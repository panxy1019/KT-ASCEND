import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import tempfile
import types
import unittest
from unittest.mock import patch


class CancelLifecycleTests(unittest.TestCase):
    def request(self, rid):
        return NS(rid=rid, finished_reason=None, to_finish=None)

    def scheduler(self, *requests):
        batch = NS(reqs=list(requests))
        return NS(
            running_batch=batch,
            cur_batch=batch,
            waiting_queue=[],
            tp_worker=NS(model_runner=NS()),
        )

    def events(self, path):
        return [json.loads(line) for line in path.read_text().splitlines()]

    def test_default_off_does_not_create_trace_or_attach_scheduler_state(self):
        from migration_bridge.round9.runtime.cancel_lifecycle import (
            http_abort_received,
            scheduler_request_admitted,
        )

        scheduler = self.scheduler(self.request("r"))
        with patch.dict(os.environ, {}, clear=True):
            http_abort_received("r", False)
            identity = scheduler_request_admitted(scheduler, scheduler.cur_batch.reqs[0], 7)

        self.assertIsNone(identity)
        self.assertFalse(hasattr(scheduler, "_round9_cancel_lifecycle"))

    def test_control_send_and_scheduler_seen_are_distinct_exact_events(self):
        from migration_bridge.round9.runtime.cancel_lifecycle import (
            http_abort_received,
            scheduler_abort_received,
            scheduler_request_admitted,
            tokenizer_abort_requested,
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cancel.jsonl"
            with patch.dict(os.environ, {
                    "ROUND9_CANCEL_TRACE_PATH": str(path),
                    "ROUND9_CANCEL_CLOSURE_ENABLE": "1"}, clear=True):
                live = self.request("live-request")
                scheduler = self.scheduler(live)
                identity = scheduler_request_admitted(scheduler, live, 9)

                http_abort_received("missing", False)
                self.assertFalse(tokenizer_abort_requested(NS(rid_to_state={}), "missing", False))
                http_abort_received("live-request", False)
                self.assertTrue(tokenizer_abort_requested(
                    NS(rid_to_state={"live-request": object()}), "live-request", False))
                matched = scheduler_abort_received(
                    scheduler, NS(rid="live-request", abort_all=False))

            self.assertEqual(identity.request_id, "live-request")
            self.assertEqual(identity.request_epoch, 0)
            self.assertEqual(identity.allocation_epoch, 9)
            self.assertEqual(matched, (identity,))
            events = self.events(path)
            self.assertEqual([event["status"] for event in events], [
                "REQUEST_ADMITTED",
                "HTTP_ABORT_RECEIVED",
                "CONTROL_NOT_SENT",
                "HTTP_ABORT_RECEIVED",
                "CONTROL_SENT",
                "SCHEDULER_ABORT_RECEIVED",
                "CANCEL_SEEN",
                "SUBMIT_FENCE_PUBLISHED",
            ])
            cancel = events[-2]
            self.assertEqual(
                (cancel["request_id"], cancel["request_epoch"], cancel["allocation_epoch"]),
                ("live-request", 0, 9),
            )
            self.assertTrue(all(type(event["event_seq"]) is int for event in events))
            self.assertTrue(all(type(event["monotonic_ns"]) is int for event in events))

    def test_cancelled_epoch_rejects_new_normal_submit_and_rid_reuse_gets_new_epoch(self):
        from migration_bridge.round9.runtime.cancel_lifecycle import (
            scheduler_abort_received,
            scheduler_before_normal_submit,
            scheduler_request_admitted,
            scheduler_request_retired,
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cancel.jsonl"
            with patch.dict(os.environ, {
                    "ROUND9_CANCEL_TRACE_PATH": str(path),
                    "ROUND9_CANCEL_CLOSURE_ENABLE": "1"}, clear=True):
                first = self.request("reused")
                scheduler = self.scheduler(first)
                first_identity = scheduler_request_admitted(scheduler, first, 3)
                scheduler_abort_received(scheduler, NS(rid="reused", abort_all=False))
                self.assertFalse(scheduler_before_normal_submit(scheduler, first, "STREAMING_CHUNK"))
                scheduler_request_retired(scheduler, first)

                second = self.request("reused")
                scheduler.running_batch.reqs[:] = [second]
                scheduler.cur_batch = scheduler.running_batch
                second_identity = scheduler_request_admitted(scheduler, second, 4)
                self.assertTrue(scheduler_before_normal_submit(scheduler, second, "STREAMING_CHUNK"))

            self.assertEqual(first_identity.request_epoch, 0)
            self.assertEqual(second_identity.request_epoch, 1)
            self.assertNotEqual(first_identity, second_identity)
            events = self.events(path)
            self.assertIn("POST_CANCEL_SUBMIT_REJECTED", [event["status"] for event in events])
            self.assertEqual(events[-1]["status"], "NORMAL_SUBMIT_ADMITTED")

    def test_trace_only_observes_but_does_not_change_submission(self):
        from migration_bridge.round9.runtime.cancel_lifecycle import (
            scheduler_abort_received,
            scheduler_batch_submit_fence,
            scheduler_before_normal_submit,
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cancel.jsonl"
            request = self.request("trace-only")
            scheduler = self.scheduler(request)
            with patch.dict(os.environ, {"ROUND9_CANCEL_TRACE_PATH": str(path)}, clear=True):
                scheduler_abort_received(scheduler, NS(rid="trace-only", abort_all=False))
                self.assertTrue(scheduler_before_normal_submit(
                    scheduler, request, "STREAMING_CHUNK"))
                self.assertIsNone(scheduler_batch_submit_fence(
                    scheduler, NS(reqs=[request])))
            self.assertEqual("POST_CANCEL_SUBMIT_OBSERVED", self.events(path)[-1]["status"])

    def test_scheduler_batch_fence_rejects_before_any_model_hook(self):
        from migration_bridge.round9.runtime.cancel_lifecycle import (
            scheduler_abort_received,
            scheduler_batch_submit_fence,
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cancel.jsonl"
            request = self.request("batch-fence")
            scheduler = self.scheduler(request)
            with patch.dict(os.environ, {
                    "ROUND9_CANCEL_TRACE_PATH": str(path),
                    "ROUND9_CANCEL_CLOSURE_ENABLE": "1"}, clear=True):
                scheduler_abort_received(
                    scheduler, NS(rid="batch-fence", abort_all=False))
                with self.assertRaisesRegex(RuntimeError, "model-submit"):
                    scheduler_batch_submit_fence(
                        scheduler, NS(reqs=[request]))
            self.assertEqual(
                "POST_CANCEL_BATCH_REJECTED", self.events(path)[-1]["status"])

    def test_closure_finalizes_cancelled_chunk_without_another_forward(self):
        from migration_bridge.round9.runtime.cancel_lifecycle import (
            scheduler_abort_finished,
            scheduler_abort_received,
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cancel.jsonl"
            request = self.request("chunked")
            request.to_finish = object()
            scheduler = self.scheduler(request)
            scheduler.running_batch = NS(reqs=[])
            scheduler.chunked_req = request
            recv = NS(rid="chunked", abort_all=False)
            with patch.dict(os.environ, {
                    "ROUND9_CANCEL_TRACE_PATH": str(path),
                    "ROUND9_CANCEL_CLOSURE_ENABLE": "1"}, clear=True), patch(
                        "migration_bridge.round9.runtime.cancel_lifecycle._finalize_chunked_request"
                    ) as finalize:
                identities = scheduler_abort_received(scheduler, recv)
                retired = scheduler_abort_finished(scheduler, recv)
            self.assertEqual(1, len(identities))
            self.assertEqual(identities, retired)
            finalize.assert_called_once_with(scheduler, request)
            self.assertEqual("CANCEL_CLEANUP_COMMITTED", self.events(path)[-1]["status"])

    def test_direct_chunk_cleanup_uses_authoritative_kv_release_and_terminal_order(self):
        from migration_bridge.round9.runtime.cancel_lifecycle import (
            _finalize_chunked_request,
        )

        order = []
        request = self.request("chunked")
        request.to_finish = object()
        request.return_logprob = True
        request.check_finished = lambda: (
            order.append("check_finished"),
            setattr(request, "finished_reason", request.to_finish),
            setattr(request, "to_finish", None),
        )
        request.finished = lambda: request.finished_reason is not None
        request.time_stats = NS(
            set_completion_time=lambda: order.append("completion_time"))
        scheduler = self.scheduler()
        scheduler.tree_cache = object()
        scheduler.chunked_req = request
        scheduler._chunked_req_scheduled_last_iter = True
        scheduler.stream_output = lambda reqs, logprobs: order.append(
            ("stream_output", reqs, logprobs))

        names = ("sglang", "sglang.srt", "sglang.srt.mem_cache",
                 "sglang.srt.mem_cache.common")
        old = {name: sys.modules.get(name) for name in names}
        for name in names:
            sys.modules[name] = types.ModuleType(name)
        sys.modules[names[-1]].release_kv_cache = lambda req, cache: order.append(
            ("release_kv_cache", req, cache))
        try:
            with patch(
                "migration_bridge.round9.runtime.framework_hooks.framework_cancel",
                side_effect=lambda owner, req: order.append(
                    ("framework_cancel", owner, req)),
            ), patch(
                "migration_bridge.round9.runtime.dynamic_boundary.scheduler_cancel",
                side_effect=lambda owner, req: order.append(
                    ("dynamic_cancel", owner, req)),
            ):
                _finalize_chunked_request(scheduler, request)
        finally:
            for name, module in old.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module

        self.assertEqual(order[0], "check_finished")
        self.assertEqual(order[1], ("framework_cancel", scheduler, request))
        self.assertEqual(order[2], ("release_kv_cache", request, scheduler.tree_cache))
        self.assertEqual(order[3], ("dynamic_cancel", scheduler, request))
        self.assertEqual(order[4], "completion_time")
        self.assertEqual(order[5], ("stream_output", [request], False))
        self.assertIsNone(scheduler.chunked_req)
        self.assertFalse(scheduler._chunked_req_scheduled_last_iter)

    def test_framework_cancel_retires_all_drained_plane_ownership(self):
        from migration_bridge.round9.runtime.framework_hooks import framework_cancel

        request = self.request("planes")
        request.finished_reason = NS()
        retired = []
        aborted = []
        prefill = NS(request=request, active=False)
        prefill._retire_cancelled = lambda: setattr(prefill, "request", None)
        ple = NS(context=None, ring=NS(budget=NS(
            retire_request=lambda rid: retired.append(rid))))
        context = NS(
            request_token=request,
            active=False,
            abort=lambda token: aborted.append(("context", token)),
        )
        streaming = NS(
            request_token=request,
            active=False,
            abort=lambda token: aborted.append(("streaming", token)),
        )
        runner = NS(
            _round9_prefill=prefill,
            _round9_ple=ple,
            _round9_ple_request=request,
            _round9_ple_credit_requests={request.rid},
            _round9_joint_host_budget=None,
            _round9_context_reference=context,
            _round9_streaming=streaming,
        )

        framework_cancel(NS(tp_worker=NS(model_runner=runner)), request)

        self.assertEqual(retired, [request.rid])
        self.assertEqual(aborted, [("context", request), ("streaming", request)])
        self.assertIsNone(prefill.request)
        self.assertIsNone(runner._round9_ple_request)
        self.assertEqual(runner._round9_ple_credit_requests, set())

    def test_framework_cancel_retires_sync_ple_request_without_credit(self):
        from migration_bridge.round9.runtime.framework_hooks import framework_cancel

        request = self.request("sync-ple")
        ple = NS(context=None, ring=NS(budget=NS(
            retire_request=lambda rid: self.fail("sync fallback has no credit"))))
        runner = NS(
            _round9_prefill=None,
            _round9_ple=ple,
            _round9_ple_request=request,
            _round9_ple_credit_requests=set(),
            _round9_joint_host_budget=None,
            _round9_context_reference=None,
            _round9_streaming=None,
        )

        framework_cancel(NS(tp_worker=NS(model_runner=runner)), request)

        self.assertIsNone(runner._round9_ple_request)
        self.assertEqual(runner._round9_ple_credit_requests, set())

    def test_framework_cancel_retires_joint_budget_credit_and_checks_baseline(self):
        from migration_bridge.round9.runtime.framework_hooks import framework_cancel

        request = self.request("joint-budget")
        retired = []
        baseline = {"allocations": 0, "active_requests": 0}
        budget = NS(
            retire_request=lambda rid: retired.append(rid),
            snapshot=lambda: baseline,
        )
        ple = NS(context=None, ring=NS(budget=NS(
            retire_request=lambda rid: self.fail("joint budget owns this credit"))))
        runner = NS(
            _round9_prefill=None,
            _round9_ple=ple,
            _round9_ple_request=request,
            _round9_ple_credit_requests={request.rid},
            _round9_joint_host_budget=budget,
            _round9_joint_budget_baseline=baseline,
            _round9_context_reference=None,
            _round9_streaming=None,
        )

        framework_cancel(NS(tp_worker=NS(model_runner=runner)), request)

        self.assertEqual(retired, [request.rid])
        self.assertEqual(runner._round9_ple_credit_requests, set())
        self.assertIsNone(runner._round9_ple_request)


if __name__ == "__main__":
    unittest.main()
