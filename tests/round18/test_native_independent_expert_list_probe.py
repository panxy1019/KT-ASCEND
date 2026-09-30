from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest

from round18_integration.native_independent_expert_list_probe import (
    allocate_independent_expert_lists,
    validate_probe_receipt,
)


class _Tensor:
    _next_pointer = 100

    def __init__(self, shape):
        self.shape = shape
        self.dtype = "torch.bfloat16"
        self.device = SimpleNamespace(type="npu")
        self._base = None
        self._pointer = _Tensor._next_pointer
        _Tensor._next_pointer += 1

    def data_ptr(self):
        return self._pointer


class _Torch:
    bfloat16 = "torch.bfloat16"

    @staticmethod
    def empty(shape, *, dtype, device):
        assert dtype == _Torch.bfloat16
        assert device == "npu:0"
        return _Tensor(shape)


class NativeIndependentExpertListProbeTests(unittest.TestCase):
    def test_direct_allocator_creates_unique_independent_projection_lists(self):
        w13, w2 = allocate_independent_expert_lists(
            _Torch, physical_slots=32, device="npu:0"
        )
        self.assertEqual(len(w13), 32)
        self.assertEqual(len(w2), 32)
        self.assertTrue(all(item.shape == (2560, 1280) for item in w13))
        self.assertTrue(all(item.shape == (640, 2560) for item in w2))
        pointers = [item.data_ptr() for item in (*w13, *w2)]
        self.assertEqual(len(pointers), len(set(pointers)))
        self.assertTrue(all(item._base is None for item in (*w13, *w2)))

    def test_source_contains_no_parent_clone_or_piecewise_path(self):
        source = Path(
            "round18_integration/native_independent_expert_list_probe.py"
        ).read_text()
        self.assertNotIn("w13_weight.data.unbind", source)
        self.assertNotIn("w2_weight.data.unbind", source)
        self.assertNotIn("PIECEWISE", source)
        self.assertNotIn("as_strided", source)
        self.assertNotIn("storage().data_ptr", source)

    def test_receipt_cannot_promote_component_graph_to_full_model(self):
        receipt = {
            "status": "PASS_EP1_INDEPENDENT_LIST_EAGER_GRAPH_COMPONENT",
            "native_runner_preserved": True,
            "tp_size": 1,
            "ep_size": 1,
            "physical_slots": 32,
            "logical_experts": 512,
            "parent_constructed": False,
            "parent_rebuilt": False,
            "eager_native_consumer": "PASS",
            "graph": {
                "scope": "COMPONENT_NPUGRAPH_ONLY",
                "full_model_graph_claim": False,
                "piecewise_used": False,
                "capture_output_rejected": True,
                "mutation_inside_replay": False,
                "replay_count": 3,
            },
        }
        validate_probe_receipt(receipt)
        receipt["graph"]["full_model_graph_claim"] = True
        with self.assertRaisesRegex(ValueError, "full-model"):
            validate_probe_receipt(receipt)


if __name__ == "__main__":
    unittest.main()
