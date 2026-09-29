from __future__ import annotations

import importlib
from types import SimpleNamespace
import unittest


def _probe():
    return importlib.import_module(
        "round16_integration.b_q32_all_layer_residency_probe"
    )


class _Device:
    type = "npu"


class _Storage:
    def __init__(self, nbytes, pointer):
        self._nbytes = nbytes
        self._pointer = pointer

    def nbytes(self):
        return self._nbytes

    def data_ptr(self):
        return self._pointer


class _Tensor:
    dtype = "torch.bfloat16"
    device = _Device()

    def __init__(self, shape, pointer, storage_bytes=None):
        self.shape = shape
        expected = 2
        for value in shape:
            expected *= value
        self._storage = _Storage(
            expected if storage_bytes is None else storage_bytes, pointer
        )

    def untyped_storage(self):
        return self._storage


class _Manager:
    global_num_experts = 512
    local_num_experts = 32

    @staticmethod
    def map_global_to_local(expert_id):
        return expert_id if expert_id < 32 else -1


def _snapshot():
    return SimpleNamespace(
        resident_logical_ids=tuple(range(32)),
        logical_to_physical=tuple(range(32)) + (-1,) * 480,
    )


def _bindings(probe, *, alias_layer=None, large_storage_layer=None):
    result = []
    current = 0
    for layer in range(48):
        w13_pointer = 1000 + 2 * layer
        if alias_layer == layer:
            w13_pointer = 1000
        w13_bytes = 32 * 1280 * 2560 * 2
        if large_storage_layer == layer:
            w13_bytes = 512 * 1280 * 2560 * 2
        owner = SimpleNamespace(
            round16_source_bound_api_identity=(
                "round16-b:source-bound-q32-routed-experts:offline-v1"
            ),
            round16_logical_expert_count=512,
            round16_physical_expert_count=32,
            round16_resident_logical_ids=tuple(range(32)),
            round16_logical_to_physical=tuple(range(32)) + (-1,) * 480,
            expert_map_manager=_Manager(),
            w13_weight=_Tensor(
                (32, 1280, 2560), w13_pointer, w13_bytes
            ),
            w2_weight=_Tensor((32, 2560, 640), 1001 + 2 * layer),
        )
        before = current
        current += probe.Q32_PAYLOAD_BYTES
        result.append(
            SimpleNamespace(
                round16_api_identity=(
                    "round16-integration:q32-preallocation-binding:v1"
                ),
                owner=owner,
                construction_method="UnquantizedFusedMoEMethod",
                memory_before_bytes=before,
                memory_after_bytes=current,
            )
        )
    return result


class Q32AllLayerResidencyProbeTests(unittest.TestCase):
    def test_accepts_48_distinct_q32_owners_and_exact_aggregate(self):
        probe = _probe()
        bindings = _bindings(probe)
        receipt = probe.validate_residency_set(
            bindings,
            _snapshot(),
            allocator_before_bytes=0,
            allocator_after_bytes=probe.AGGREGATE_PAYLOAD_BYTES,
            allocator_peak_bytes=probe.AGGREGATE_PAYLOAD_BYTES,
        )
        self.assertEqual(receipt["status"], "ALL_LAYER_RESIDENCY_VERIFIED")
        self.assertEqual(receipt["layer_count"], 48)
        self.assertEqual(receipt["distinct_owner_count"], 48)
        self.assertEqual(receipt["distinct_storage_count"], 96)
        self.assertEqual(
            receipt["nominal_payload_bytes"], 15_099_494_400
        )

    def test_rejects_cross_layer_alias_and_hidden_full512_storage(self):
        probe = _probe()
        for bindings, message in (
            (_bindings(probe, alias_layer=1), "aliases another layer"),
            (_bindings(probe, large_storage_layer=1), "larger storage"),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    probe.validate_residency_set(
                        bindings,
                        _snapshot(),
                        allocator_before_bytes=0,
                        allocator_after_bytes=probe.AGGREGATE_PAYLOAD_BYTES,
                        allocator_peak_bytes=probe.AGGREGATE_PAYLOAD_BYTES,
                    )

    def test_requires_complete_set_and_explicit_device_enable(self):
        probe = _probe()
        with self.assertRaisesRegex(ValueError, "exactly 48"):
            probe.validate_residency_set(
                _bindings(probe)[:-1],
                _snapshot(),
                allocator_before_bytes=0,
                allocator_after_bytes=probe.AGGREGATE_PAYLOAD_BYTES,
                allocator_peak_bytes=probe.AGGREGATE_PAYLOAD_BYTES,
            )
        with self.assertRaisesRegex(RuntimeError, "ROUND16_B_ENABLE_Q32_V4"):
            probe.require_device_enable({})


if __name__ == "__main__":
    unittest.main()
