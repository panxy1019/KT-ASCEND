import json
import os
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
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
            with patch.dict(os.environ, {"ROUND9_CANCEL_TRACE_PATH": str(path)}, clear=True):
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
            with patch.dict(os.environ, {"ROUND9_CANCEL_TRACE_PATH": str(path)}, clear=True):
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


if __name__ == "__main__":
    unittest.main()
