"""CPU-only contracts for the bounded asynchronous PLE row ring."""
from concurrent.futures import Future
import unittest


class FakeReader:
    width = 2
    total_rows = 8

    def gather_bytes(self, row_ids):
        return b"".join(bytes((row, row, row, row)) for row in row_ids)


class FakeEvent:
    def __init__(self, complete=False):
        self.complete = complete


class FakeBackend:
    def __init__(self):
        self.allocated = []
        self.pending = []
        self.launched = []
        self.waited = []
        self.recorded = []
        self.synchronized = []
        self.released = []
        self.closed = False

    def allocate_slots(self, count, max_rows, row_width):
        self.allocated.append((count, max_rows, row_width))
        return [dict(index=i, raw=None) for i in range(count)]

    def submit_gather(self, reader, row_ids, slot, identity):
        future = Future()
        self.pending.append((future, reader, tuple(row_ids), slot, identity))
        return future

    def finish_gather(self, index=0, *, identity=None, error=None):
        future, reader, row_ids, slot, expected = self.pending[index]
        if error is not None:
            future.set_exception(error)
            return
        slot["raw"] = reader.gather_bytes(row_ids)
        future.set_result(expected if identity is None else identity)

    def launch_h2d(self, slot, row_count, row_shape, identity):
        event = FakeEvent()
        result = (identity, tuple(row_shape), slot["raw"])
        self.launched.append((slot["index"], row_count, tuple(row_shape), identity, event))
        return result, event

    def query_event(self, event, identity):
        return event.complete

    def wait_event(self, event, consumer_stream, identity):
        self.waited.append((event, consumer_stream, identity))
        return consumer_stream

    def record_event(self, consumer_stream, identity):
        event = FakeEvent()
        self.recorded.append((event, consumer_stream, identity))
        return event

    def synchronize_event(self, event, identity):
        self.synchronized.append((event, identity))
        event.complete = True

    def release_slot(self, slot, identity):
        self.released.append((slot["index"], identity))
        slot["raw"] = None

    def close(self):
        self.closed = True


class AsyncPLERingTests(unittest.TestCase):
    def api(self):
        from migration_bridge.round9.runtime import ple_async
        return ple_async

    def budget(self, bytes_=128):
        from migration_bridge.round9.runtime.host_budget import HostWorkBudget
        return HostWorkBudget(
            enabled=True,
            host_bytes=bytes_,
            h2d_bytes=bytes_,
            pinned_bytes=bytes_,
        )

    def ring(self, *, backend=None, budget=None, max_rows=4):
        api = self.api()
        return api.AsyncPLERing(
            FakeReader(), enabled=True, max_rows=max_rows,
            budget=self.budget() if budget is None else budget,
            backend=FakeBackend() if backend is None else backend,
        )

    def test_default_off_does_not_allocate_or_replace_sync_path(self):
        api = self.api()
        backend = FakeBackend()
        ring = api.AsyncPLERing(FakeReader(), backend=backend)

        self.assertEqual(backend.allocated, [])
        self.assertIsNone(ring.submit_exact(
            "request", 0, [[1, 2]], phase=api.PREFILL_INPUT,
        ))
        self.assertEqual(ring.snapshot()["state"], "OFF")

    def test_exact_rows_preserve_duplicates_shape_and_allocation_identity(self):
        api = self.api()
        backend = FakeBackend()
        budget = self.budget()
        ring = self.ring(backend=backend, budget=budget)

        ticket = ring.submit_exact(
            "request-a", 7, [[7, 0], [4, 4]], phase=api.PREFILL_INPUT,
        )
        self.assertEqual(
            (ticket.identity.request_id, ticket.identity.placement_epoch,
             ticket.identity.allocation_epoch),
            ("request-a", 7, 0),
        )
        self.assertEqual(backend.pending[0][2], (7, 0, 4, 4))
        self.assertEqual(
            budget.snapshot()["used"],
            {"host_bytes": 16, "h2d_bytes": 16, "pinned_bytes": 16},
        )

        backend.finish_gather()
        result = ring.wait(ticket, consumer_stream="model-stream")
        self.assertEqual(result[0], ticket.identity)
        self.assertEqual(result[1], (2, 2))
        self.assertEqual(
            result[2],
            bytes((7, 7, 7, 7, 0, 0, 0, 0, 4, 4, 4, 4, 4, 4, 4, 4)),
        )
        self.assertEqual(backend.waited[0][1:], ("model-stream", ticket.identity))

        ring.release(ticket)
        self.assertEqual(backend.synchronized[0][1], ticket.identity)
        self.assertEqual(budget.snapshot()["used"], {
            "host_bytes": 0, "h2d_bytes": 0, "pinned_bytes": 0,
        })

    def test_two_slots_are_bounded_and_current_work_falls_back_while_cancel_drains(self):
        api = self.api()
        backend = FakeBackend()
        ring = self.ring(backend=backend)
        first = ring.submit_exact(
            "prefetch-a", 0, [[1]], phase=api.PREFILL_INPUT,
            priority=api.Priority.PLE_PREFETCH,
        )
        second = ring.submit_exact(
            "prefetch-b", 0, [[2]], phase=api.PREFILL_INPUT,
            priority=api.Priority.PLE_PREFETCH,
        )
        self.assertIsNone(ring.submit_exact(
            "current", 0, [[3]], phase=api.SAMPLED_DECODE,
            priority=api.Priority.PLE_CURRENT,
        ))
        self.assertEqual(ring.snapshot()["cancelled_pending"], 2)

        # A cancelled gather owns its pinned slot until its late completion is
        # observed. It can then be retired, but its old ticket stays stale.
        backend.finish_gather(0)
        self.assertEqual(ring.poll(first), "RETIRED")
        current = ring.submit_exact(
            "current", 0, [[3]], phase=api.SAMPLED_DECODE,
            priority=api.Priority.PLE_CURRENT,
        )
        self.assertIsNotNone(current)
        self.assertNotEqual(current.identity, first.identity)
        with self.assertRaisesRegex(ValueError, "stale"):
            ring.poll(first)

        backend.finish_gather(1)
        self.assertEqual(ring.poll(second), "RETIRED")

    def test_cancelled_transfer_waits_for_event_before_slot_reuse(self):
        api = self.api()
        backend = FakeBackend()
        ring = self.ring(backend=backend)
        old = ring.submit_exact(
            "old", 2, [[1, 2]], phase=api.PREFILL_INPUT,
        )
        backend.finish_gather()
        ring.wait(old)

        ring.cancel(old)
        self.assertEqual(backend.recorded[0][1], None)
        self.assertIs(backend.synchronized[0][0], backend.recorded[0][0])
        self.assertEqual(backend.synchronized[0][1], old.identity)
        self.assertEqual(backend.released[0][1], old.identity)
        new = ring.submit_exact(
            "new", 3, [[3, 4]], phase=api.PREFILL_INPUT,
        )
        self.assertEqual(backend.pending[-1][3]["index"], backend.pending[0][3]["index"])
        self.assertGreater(new.identity.allocation_epoch, old.identity.allocation_epoch)

    def test_current_admission_never_blocks_on_optional_transfer_event(self):
        api = self.api()
        backend = FakeBackend()
        ring = self.ring(backend=backend)
        optional = []
        for index in range(2):
            ticket = ring.submit_exact(
                f"optional-{index}", 0, [[index]], phase=api.PREFILL_INPUT,
                priority=api.Priority.PLE_PREFETCH,
            )
            backend.finish_gather(index)
            self.assertEqual(ring.poll(ticket), "TRANSFERRING")
            optional.append(ticket)

        self.assertIsNone(ring.submit_exact(
            "current", 0, [[3]], phase=api.SAMPLED_DECODE,
            priority=api.Priority.PLE_CURRENT,
        ))
        self.assertEqual(backend.synchronized, [])
        self.assertEqual(ring.snapshot()["cancelled_pending"], 2)

        backend.launched[0][-1].complete = True
        self.assertEqual(ring.poll(optional[0]), "RETIRED")
        self.assertIsNotNone(ring.submit_exact(
            "current", 0, [[3]], phase=api.SAMPLED_DECODE,
            priority=api.Priority.PLE_CURRENT,
        ))

    def test_invalid_or_inferred_rows_are_rejected_before_admission(self):
        api = self.api()
        backend = FakeBackend()
        budget = self.budget()
        ring = self.ring(backend=backend, budget=budget, max_rows=3)
        bad = (
            ([[1]], "future_decode"),
            ([[1], [2, 3]], api.PREFILL_INPUT),
            ([[1, 2], [3, 4]], api.PREFILL_INPUT),
            ([[8]], api.PREFILL_INPUT),
            ([[True]], api.PREFILL_INPUT),
            ([], api.PREFILL_INPUT),
        )
        for rows, phase in bad:
            with self.subTest(rows=rows, phase=phase), self.assertRaises(ValueError):
                ring.submit_exact("request", 0, rows, phase=phase)
        self.assertEqual(backend.pending, [])
        self.assertEqual(budget.snapshot()["allocations"], 0)

    def test_gather_error_and_wrong_late_identity_retire_without_reuse_leak(self):
        api = self.api()
        backend = FakeBackend()
        budget = self.budget()
        ring = self.ring(backend=backend, budget=budget)
        failed = ring.submit_exact("a", 0, [[1]], phase=api.PREFILL_INPUT)
        backend.finish_gather(error=OSError("short row"))
        with self.assertRaisesRegex(OSError, "short row"):
            ring.poll(failed)
        self.assertEqual(budget.snapshot()["allocations"], 0)

        stale = ring.submit_exact("b", 1, [[2]], phase=api.PREFILL_INPUT)
        wrong = api.BufferIdentity("b", 1, stale.identity.allocation_epoch + 99)
        backend.finish_gather(1, identity=wrong)
        with self.assertRaisesRegex(RuntimeError, "identity"):
            ring.poll(stale)
        self.assertEqual(budget.snapshot()["allocations"], 0)

    def test_close_drains_cpu_and_device_work_then_closes_backend(self):
        api = self.api()
        backend = FakeBackend()
        ring = self.ring(backend=backend)
        ticket = ring.submit_exact("a", 0, [[1]], phase=api.PREFILL_INPUT)
        backend.finish_gather()
        ring.wait(ticket)
        ring.close()
        self.assertEqual(backend.synchronized[0][1], ticket.identity)
        self.assertTrue(backend.closed)
        self.assertEqual(ring.snapshot()["state"], "CLOSED")


if __name__ == "__main__":
    unittest.main()
