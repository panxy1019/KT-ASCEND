from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from round16.a.fresh_a1_profile import BLOCK_IDS
from round16.a import fresh_a1_runtime_observer as observer


class _Model:
    def __init__(self, method: object) -> None:
        self.method = method
        self.modules = []
        for layer in range(48):
            value = method if layer == 24 else _ktep_type()
            self.modules.append(types.SimpleNamespace(quant_method=value))

    def named_modules(self):
        return tuple(
            (f"model.layers.{layer}.mlp.experts", module)
            for layer, module in enumerate(self.modules)
        )


class _Runner:
    def __init__(self, method: object) -> None:
        self.model_runner = types.SimpleNamespace(model=_Model(method))


def _ktep_type(*, fail: BaseException | None = None):
    def submit(self, value):
        self.events.append(("submit", value))
        return None

    def sync(self, value):
        self.events.append(("sync", value))
        return value + 1

    def apply(self, value):
        self.events.append(("apply", value))
        self._submit_cpu_forward(value)
        result = self._sync_cpu_forward(value)
        if fail is not None:
            raise fail
        return result

    cls = type(
        "KTEPWrapperMethod",
        (),
        {
            "_submit_cpu_forward": submit,
            "_sync_cpu_forward": sync,
            "apply": apply,
        },
    )
    value = cls()
    value.events = []
    value.kt_config = types.SimpleNamespace(layer_idx=24)
    return value


class FreshA1RuntimeObserverTests(unittest.TestCase):
    def tearDown(self) -> None:
        observer._reset_for_tests()

    def _environment(self, profile: Path) -> dict[str, str]:
        return {
            "ROUND16_A1_PROFILE_ENABLE": "1",
            "ROUND16_A1_BLOCK_ID": BLOCK_IDS["A"],
            "ROUND16_A1_PROFILE_PATH": str(profile),
            "ROUND16_A1_BINDINGS_PATH": str(profile.with_name("bindings.json")),
        }

    def test_import_is_side_effect_free_and_does_not_import_torch(self) -> None:
        code = (
            "import sys; "
            "import round16.a.fresh_a1_runtime_observer; "
            "assert 'torch' not in sys.modules"
        )
        env = dict(os.environ)
        env.pop("ROUND16_A1_PROFILE_ENABLE", None)
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=Path(__file__).resolve().parents[2],
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_default_off_paths_are_transparent_and_do_not_inspect_runner(self) -> None:
        class ForbiddenRunner:
            def __getattribute__(self, name):
                raise AssertionError(f"default-off inspected runner.{name}")

        expected = object()
        error = LookupError("same object")

        with patch.dict(os.environ, {}, clear=True):
            observer.before(ForbiddenRunner(), object())
            self.assertIs(observer.observe_host_merge(24, lambda: expected), expected)
            self.assertIs(observer.observe_npu_call("NPU_RESIDENT", 24, lambda: expected), expected)
            with self.assertRaises(LookupError) as raised:
                observer.observe_host_merge(24, lambda: (_ for _ in ()).throw(error))
            observer.after(ForbiddenRunner())

        self.assertIs(raised.exception, error)

    def test_enabled_binding_records_parent_host_bookends_and_explicit_cpu_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as item:
            profile = Path(item) / "profile.jsonl"
            method = _ktep_type()
            runner = _Runner(method)
            with patch.dict(os.environ, self._environment(profile), clear=True):
                observer.before(runner, [object()])
                result = method.apply(7)
                observer.after(runner)

            self.assertEqual(result, 8)
            rows = [json.loads(line) for line in profile.read_text().splitlines()]
            components = {row["component"] for row in rows}
            self.assertEqual(
                components,
                {"PARENT_ROUTED_EXPERT", "HOST_DISPATCH", "HOST_SYNC", "CPU_EXPERT"},
            )
            unknown = next(row for row in rows if row["component"] == "CPU_EXPERT")
            parent = next(
                row for row in rows if row["component"] == "PARENT_ROUTED_EXPERT"
            )
            self.assertEqual(unknown["status"], "UNKNOWN_NATIVE_ABI_NO_CALLBACK")
            bindings = json.loads(profile.with_name("bindings.json").read_text())
            self.assertEqual(bindings["host_layers"], list(range(48)))
            self.assertEqual(bindings["device_sample_layers"], [0, 24, 47])
            self.assertEqual(unknown["clock_domain"], "NATIVE_WORKER_CLOCK_UNAVAILABLE")
            self.assertIsNone(unknown["start_ns"])
            self.assertIsNone(unknown["end_ns"])
            self.assertIsNone(unknown["duration_ns"])
            host = next(row for row in rows if row["component"] == "HOST_DISPATCH")
            self.assertEqual(host["clock_domain"], "HOST_MONOTONIC_NS")
            self.assertEqual(host["parent_span_id"], parent["span_id"])
            self.assertEqual(host["correlation_id"], parent["correlation_id"])
            self.assertEqual(unknown["parent_span_id"], parent["span_id"])
            for field in (
                "span_id",
                "parent_span_id",
                "stream_id",
                "start_event_id",
                "end_event_id",
                "correlation_id",
            ):
                self.assertIsInstance(host[field], str)
                self.assertTrue(host[field])

    def test_actual_scheduler_tp_worker_model_runner_path_is_bound(self) -> None:
        with tempfile.TemporaryDirectory() as item:
            profile = Path(item) / "profile.jsonl"
            method = _ktep_type()
            scheduler = types.SimpleNamespace(
                tp_worker=types.SimpleNamespace(model_runner=_Runner(method).model_runner)
            )
            with patch.dict(os.environ, self._environment(profile), clear=True):
                observer.before(scheduler, [object()])
                self.assertEqual(method.apply(10), 11)
                observer.after(scheduler)

            rows = [json.loads(line) for line in profile.read_text().splitlines()]
            self.assertIn("PARENT_ROUTED_EXPERT", {row["component"] for row in rows})

    def test_actual_batch_request_identity_is_emitted(self) -> None:
        with tempfile.TemporaryDirectory() as item:
            profile = Path(item) / "profile.jsonl"
            method = _ktep_type()
            scheduler = types.SimpleNamespace(
                tp_worker=types.SimpleNamespace(model_runner=_Runner(method).model_runner)
            )
            batch = types.SimpleNamespace(
                reqs=[types.SimpleNamespace(rid="round16-a-request-3")]
            )
            with patch.dict(os.environ, self._environment(profile), clear=True):
                observer.before(scheduler, batch)
                method.apply(2)
                observer.after(scheduler)

            rows = [json.loads(line) for line in profile.read_text().splitlines()]
            self.assertTrue(rows)
            self.assertEqual(
                {row["request_id"] for row in rows}, {"round16-a-request-3"}
            )

    def test_wrapped_parent_preserves_exception_identity_and_flushes_error_rows(self) -> None:
        with tempfile.TemporaryDirectory() as item:
            profile = Path(item) / "profile.jsonl"
            expected = RuntimeError("same object")
            method = _ktep_type(fail=expected)
            runner = _Runner(method)
            with patch.dict(os.environ, self._environment(profile), clear=True):
                observer.before(runner, [object()])
                with self.assertRaises(RuntimeError) as raised:
                    method.apply(4)
                observer.after(runner)

            self.assertIs(raised.exception, expected)
            rows = [json.loads(line) for line in profile.read_text().splitlines()]
            parent = next(row for row in rows if row["component"] == "PARENT_ROUTED_EXPERT")
            self.assertEqual(parent["outcome"], "error")

    def test_enabled_npu_event_records_deferred_elapsed_without_synchronize(self) -> None:
        with tempfile.TemporaryDirectory() as item:
            profile = Path(item) / "profile.jsonl"
            calls: list[str] = []

            class Event:
                def __init__(self, enable_timing=False):
                    self.enable_timing = enable_timing

                def record(self):
                    calls.append("record")

                def query(self):
                    calls.append("query")
                    return True

                def elapsed_time(self, other):
                    calls.append("elapsed_time")
                    return 1.25

            fake_npu = types.SimpleNamespace(
                Event=Event,
                current_stream=lambda: "fake-npu-stream-0",
            )
            fake_torch = types.SimpleNamespace(npu=fake_npu)
            runner = _Runner(_ktep_type())
            with (
                patch.dict(os.environ, self._environment(profile), clear=True),
                patch.dict(sys.modules, {"torch": fake_torch}),
            ):
                observer.before(runner, [object()])
                result = observer.observe_npu_call("NPU_RESIDENT", 24, lambda value: value + 2, 5)
                observer.after(runner)

            self.assertEqual(result, 7)
            self.assertNotIn("synchronize", calls)
            rows = [json.loads(line) for line in profile.read_text().splitlines()]
            device = next(row for row in rows if row["component"] == "NPU_RESIDENT")
            self.assertEqual(device["clock_domain"], "NPU_EVENT_ELAPSED_MS")
            self.assertEqual(device["elapsed_ms"], 1.25)
            self.assertEqual(device["query_policy"], "DEFERRED_QUERY_NO_SYNCHRONIZE")
            self.assertIsNone(device["start_ns"])
            self.assertIsNone(device["end_ns"])

    def test_not_ready_device_event_is_retained_and_flushed_by_later_batch(self) -> None:
        with tempfile.TemporaryDirectory() as item:
            profile = Path(item) / "profile.jsonl"
            ready = [False]
            calls: list[str] = []

            class Event:
                def __init__(self, enable_timing=False):
                    pass

                def record(self):
                    calls.append("record")

                def query(self):
                    calls.append("query")
                    return ready[0]

                def elapsed_time(self, other):
                    calls.append("elapsed_time")
                    return 2.5

            fake_torch = types.SimpleNamespace(
                npu=types.SimpleNamespace(Event=Event, current_stream=lambda: "npu:0")
            )
            runner = _Runner(_ktep_type())
            with (
                patch.dict(os.environ, self._environment(profile), clear=True),
                patch.dict(sys.modules, {"torch": fake_torch}),
            ):
                observer.before(runner, [object()])
                observer.observe_npu_call("NPU_RESIDENT", 24, lambda: None)
                observer.after(runner)
                self.assertNotIn("elapsed_time", calls)
                ready[0] = True
                observer.before(runner, [object()])
                observer.after(runner)

            rows = [json.loads(line) for line in profile.read_text().splitlines()]
            device = [row for row in rows if row["component"] == "NPU_RESIDENT"]
            self.assertEqual(len(device), 1)
            self.assertEqual(device[0]["status"], "COMPLETE")
            self.assertEqual(device[0]["elapsed_ms"], 2.5)
            self.assertNotIn("synchronize", calls)


if __name__ == "__main__":
    unittest.main()
