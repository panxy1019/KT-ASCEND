"""Independent enabled-path regressions for the request-cost observer."""
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import unittest
from unittest.mock import patch

from migration_bridge.round9.runtime import framework_hooks, layerwise
from migration_bridge.round9.runtime import request_cost_binding as binding


class _Adapter:
    def __init__(self):
        self.copied_bytes = 0

    def copy_experts(self, layer, moves):
        self.copied_bytes += len(moves) * 64
        return (layer, tuple(moves))


class _Method:
    def apply(self, layer, dispatch):
        return (layer, dispatch)


class _Backend:
    def __init__(self):
        self.adapter = _Adapter()
        self.method = _Method()

    def execute_group(self, layer, group, hidden, weights, identity):
        self.adapter.copy_experts(
            layer, tuple(enumerate(group.logical_experts)))
        self.method.apply(layer, "dispatch")
        return hidden

    def synchronize(self, identity):
        return identity

    def release(self, identity):
        return identity

    def assemble(self, rows, hidden):
        return hidden


class _Runtime:
    def __init__(self):
        self.transport = NS(backend=_Backend())

    @staticmethod
    def _rows(value):
        return value

    def apply(self, layer, hidden, ids, weights):
        self._rows(ids)
        self._rows(weights)
        group = layerwise.group_topk_routes(
            ids, capacity=2, experts=4, topk=2)[0]
        identity = NS(allocation_epoch=19)
        values = self.transport.backend.execute_group(
            layer, group, hidden, weights, identity)
        self.transport.backend.synchronize(identity)
        self.transport.backend.release(identity)
        return self.transport.backend.assemble(values, hidden)


class _Mode:
    def is_extend(self):
        return True

    def is_decode(self):
        return False


class _Request:
    def __init__(self, rid="request-1"):
        self.rid = rid
        self.origin_input_ids = [7]
        self.last_extend_prefix_len = 0
        self.extend_input_len = 1
        self.finished_reason = None
        self.done = False

    def finished(self):
        return self.done


class RequestCostEnabledBindingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.output = self.directory / "request-cost.jsonl"
        self.profile = self.directory / "candidate.json"
        self.profile.write_text("{}\n")
        runtime_names = (
            "layerwise.py", "prefill_runtime.py", "prefill_binding.py",
            "framework_hooks.py", "request_cost_binding.py",
            "request_cost_ledger.py",
        )
        runtime_files = {
            name: hashlib.sha256(Path(binding.__file__).with_name(name).read_bytes()).hexdigest()
            for name in runtime_names
        }
        runtime_sha256 = hashlib.sha256(json.dumps(
            runtime_files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.manifest = self.directory / "request-cost-manifest.json"
        self.manifest.write_text(json.dumps({
            "schema": 1,
            "runtime_files": runtime_files,
            "runtime_sha256": runtime_sha256,
            "profile_sha256": hashlib.sha256(self.profile.read_bytes()).hexdigest(),
            "max_events": 20000,
        }, sort_keys=True))
        self.environment = {
            "ROUND9_REQUEST_COST_ENABLE": "1",
            "ROUND9_REQUEST_COST_MANIFEST": str(self.manifest),
            "ROUND9_REQUEST_COST_MANIFEST_SHA256": hashlib.sha256(
                self.manifest.read_bytes()).hexdigest(),
            "ROUND9_CANDIDATE_PROFILE": str(self.profile),
        }

    def _runner(self):
        args = NS(max_running_requests=1, disable_overlap_schedule=True,
                  disable_radix_cache=True, tp_size=1, kt_cpuinfer=128,
                  kt_num_gpu_experts=32)
        context = NS(identity=NS(allocation_epoch=23))
        return NS(server_args=args, _round9_prefill=_Runtime(),
                  _round9_context_reference=context)

    def _bind(self, run_batch, process_batch_result=None):
        runner = self._runner()
        scheduler = NS(
            tp_worker=NS(model_runner=runner),
            run_batch=run_batch,
            process_batch_result=(process_batch_result or
                                  (lambda batch, result: result)),
        )
        real_recorder = binding.Recorder
        recorder_factory = lambda unused, identity, **kwargs: real_recorder(
            self.output, identity, **kwargs)
        patches = (
            patch.object(binding, "Recorder", side_effect=recorder_factory),
            patch.object(framework_hooks, "framework_before",
                         side_effect=lambda scheduler, batch: None),
            patch.object(framework_hooks, "framework_after",
                         side_effect=lambda scheduler: None),
            patch.object(framework_hooks, "framework_cancel",
                         side_effect=lambda scheduler, request: None),
            patch.object(layerwise, "group_topk_routes",
                         wraps=layerwise.group_topk_routes),
            patch.dict(os.environ, self.environment, clear=True),
        )
        for item in patches:
            item.start()
            self.addCleanup(item.stop)
        self.assertTrue(binding.bind(runner))
        return runner, scheduler

    def test_enabled_bind_records_scheduler_model_and_group_stages(self):
        runner, scheduler = self._bind(
            lambda batch: runner._round9_prefill.apply(
                3, [[1.0]], [[0, 1]], [[0.6, 0.4]]))
        request = _Request()
        batch = NS(reqs=[request], forward_mode=_Mode())

        framework_hooks.framework_before(scheduler, batch)
        result = scheduler.run_batch(batch)
        scheduler.process_batch_result(batch, result)
        request.done = True
        framework_hooks.framework_after(scheduler)

        document = json.loads(self.output.read_text())
        self.assertEqual(document["status"], "COMPLETE_HOST_ACCOUNTING")
        self.assertEqual(document["termination"], "finished")
        self.assertEqual(document["identity"]["request_allocation_epoch"], 23)
        stages = [row["stage"] for row in document["events"]]
        self.assertIn("model_forward", stages)
        self.assertIn("result_processing", stages)
        self.assertEqual(document["totals"]["prepare"]["logical_bytes"], 128)
        for stage in ("expert_group", "prepare", "compute_enqueue",
                      "device_wait", "release"):
            rows = [row for row in document["events"] if row["stage"] == stage]
            self.assertEqual(len(rows), 1, stage)
            self.assertEqual(rows[0]["group"], 19, stage)
        group = next(row for row in document["events"]
                     if row["stage"] == "expert_group")
        self.assertEqual(group["token_rows"], 1)
        self.assertEqual(group["expert_count"], 2)

    def test_run_batch_exception_identity_is_preserved_and_flushed_invalid(self):
        provider_error = RuntimeError("provider failure")

        def fail(batch):
            raise provider_error

        unused_runner, scheduler = self._bind(fail)
        request = _Request()
        batch = NS(reqs=[request], forward_mode=_Mode())
        framework_hooks.framework_before(scheduler, batch)

        with self.assertRaises(RuntimeError) as caught:
            scheduler.run_batch(batch)
        self.assertIs(caught.exception, provider_error)
        document = json.loads(self.output.read_text())
        self.assertEqual(document["status"], "INVALID_EVIDENCE")
        self.assertEqual(document["termination"], "provider_error")
        self.assertIn("PROVIDER_ERROR", document["errors"])
        self.assertEqual(document["totals"], {})

    def test_result_processing_exception_uses_the_same_failure_path(self):
        provider_error = LookupError("result processing failure")

        def fail(batch, result):
            raise provider_error

        unused_runner, scheduler = self._bind(lambda batch: "result", fail)
        request = _Request()
        batch = NS(reqs=[request], forward_mode=_Mode())
        framework_hooks.framework_before(scheduler, batch)
        result = scheduler.run_batch(batch)

        with self.assertRaises(LookupError) as caught:
            scheduler.process_batch_result(batch, result)
        self.assertIs(caught.exception, provider_error)
        document = json.loads(self.output.read_text())
        self.assertEqual(document["status"], "INVALID_EVIDENCE")
        self.assertIn("PROVIDER_ERROR", document["errors"])

    def test_observer_fsync_failure_does_not_replace_provider_exception(self):
        provider_error = RuntimeError("provider survives observer I/O")

        def fail(batch):
            raise provider_error

        unused_runner, scheduler = self._bind(fail)
        request = _Request()
        batch = NS(reqs=[request], forward_mode=_Mode())
        framework_hooks.framework_before(scheduler, batch)

        with patch.object(binding.os, "fsync", side_effect=OSError("disk")):
            with self.assertRaises(RuntimeError) as caught:
                scheduler.run_batch(batch)
        self.assertIs(caught.exception, provider_error)


if __name__ == "__main__":
    unittest.main()
