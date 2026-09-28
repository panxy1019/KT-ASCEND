import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from round14a_five_bucket import (
    StorageLedger,
    classify_model_name,
    summarize_allocators,
)


class FakeStorage:
    def __init__(self, pointer, size):
        self.pointer = pointer
        self.size = size

    def data_ptr(self):
        return self.pointer

    def nbytes(self):
        return self.size


class FakeTensor:
    def __init__(self, pointer, size, device="npu:0", shape=(1,), dtype="bf16"):
        self._storage = FakeStorage(pointer, size)
        self.device = device
        self.shape = shape
        self.dtype = dtype

    def untyped_storage(self):
        return self._storage

    def numel(self):
        return 1

    def element_size(self):
        return 2


class FiveBucketTests(unittest.TestCase):
    def test_model_name_classification(self):
        self.assertEqual(classify_model_name("model.layers.0.mlp.experts.w13_weight"), "Expert")
        self.assertEqual(classify_model_name("model.layers.1.ple.table"), "PLE")
        self.assertEqual(classify_model_name("model.layers.2.gdn.in_proj.weight"), "Dense backbone")

    def test_storage_alias_counted_once_and_conflict_reported(self):
        ledger = StorageLedger()
        first = FakeTensor(1234, 4096)
        alias = FakeTensor(1234, 4096)
        ledger.add("Expert", "expert.weight", first)
        ledger.add("Workspace", "temporary.view", alias)
        result = ledger.document()
        self.assertEqual(result["buckets"]["Expert"]["npu_bytes"], 4096)
        self.assertEqual(result["buckets"]["Workspace"]["npu_bytes"], 0)
        self.assertEqual(result["unique_storage_count"], 1)
        self.assertEqual(result["aliases"][0]["owner_bucket"], "Expert")
        self.assertEqual(result["aliases"][0]["alias_bucket"], "Workspace")

    def test_allocator_residual_is_not_assigned_to_persistent_bucket(self):
        result = summarize_allocators(
            allocated=1000,
            reserved=1400,
            persistent_npu_bytes=700,
            board_used=1800,
            owned_used=1200,
        )
        self.assertEqual(result["workspace_unclassified_allocated_bytes"], 300)
        self.assertEqual(result["allocator_reserved_unallocated_bytes"], 400)
        self.assertEqual(result["board_driver_or_external_residual_bytes"], 600)


if __name__ == "__main__":
    unittest.main()
