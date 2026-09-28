import os
import unittest

from round14a_concurrency_runtime import (
    BatchedContextReferenceRuntime, request_slices, validate_scheduler_requests,
)


class Values:
    def __init__(self, values): self.values = values
    def detach(self): return self
    def cpu(self): return self
    def tolist(self): return list(self.values)


class Mode:
    def __init__(self, name): self.name = name
    def is_decode(self): return self.name == "decode"
    def is_extend(self): return self.name == "extend"


class Batch:
    def __init__(self, mode, slots, lengths=None):
        self.batch_size = len(slots)
        self.req_pool_indices = Values(slots)
        self.extend_seq_lens_cpu = lengths
        self.forward_mode = Mode(mode)


class Request:
    def __init__(self, rid, slot, total):
        self.rid, self.req_pool_idx = rid, slot
        self.origin_input_ids = list(range(total))


class SchedulerBatch:
    def __init__(self, requests): self.reqs = requests


class Context:
    def __init__(self):
        self.request_token = None
        self.next_layer = 0
        self.active = False
        self.after_calls = []
    def begin(self, rid, request):
        self.request_token = request; self.active = True
    def after(self, request, *, finished):
        self.after_calls.append((request, finished)); self.active = False
        if finished: self.request_token = None
    def consume(self, *args, **kwargs): return "consumed"


class FinishingRequest(Request):
    def __init__(self, rid, slot, total):
        super().__init__(rid, slot, total); self.done = False
    def finished(self): return self.done


class ConcurrencyRuntimeTests(unittest.TestCase):
    def setUp(self): os.environ["ROUND14A_CONCURRENCY_MAX"] = "4"

    def tearDown(self): os.environ.pop("ROUND14A_CONCURRENCY_MAX", None)

    def test_decode_has_one_row_per_request(self):
        self.assertEqual(request_slices(Batch("decode", [7, 9, 11]), 3),
                         ((0, 7, 0, 1), (1, 9, 1, 2), (2, 11, 2, 3)))

    def test_extend_uses_exact_lengths(self):
        self.assertEqual(request_slices(Batch("extend", [7, 9], [3, 5]), 8),
                         ((0, 7, 0, 3), (1, 9, 3, 8)))

    def test_span_or_slot_mismatch_is_rejected(self):
        with self.assertRaises(RuntimeError):
            request_slices(Batch("extend", [7, 9], [3, 5]), 7)
        with self.assertRaises(RuntimeError):
            request_slices(Batch("decode", [7, 7]), 2)

    def test_h129_is_checked_per_request(self):
        batch = SchedulerBatch([Request("a", 3, 128), Request("b", 4, 64)])
        rows = validate_scheduler_requests(batch)
        self.assertEqual([(row[0], row[2]) for row in rows], [(3, 128), (4, 64)])
        with self.assertRaises(RuntimeError):
            validate_scheduler_requests(SchedulerBatch([Request("x", 8, 129)]))

    def test_after_uses_frozen_slot_after_scheduler_clears_request_field(self):
        made = []
        def factory():
            value = Context(); made.append(value); return value
        runtime = BatchedContextReferenceRuntime(factory)
        request = FinishingRequest("a", 7, 64)
        runtime.begin_batch(SchedulerBatch([request]))
        request.done = True
        request.req_pool_idx = None
        runtime.after_batch()
        self.assertFalse(runtime.active)
        self.assertEqual(made[0].after_calls, [(request, True)])


if __name__ == "__main__": unittest.main()
