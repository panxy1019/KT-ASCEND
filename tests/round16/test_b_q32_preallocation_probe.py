from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest


def _probe():
    return importlib.import_module("round16_integration.b_q32_preallocation_probe")


class _Device:
    type = "npu"

    def __str__(self) -> str:
        return "npu:0"


class _Storage:
    def __init__(self, nbytes: int) -> None:
        self._nbytes = nbytes

    def nbytes(self) -> int:
        return self._nbytes


class _Tensor:
    def __init__(self, shape, *, storage_bytes=None) -> None:
        self.shape = tuple(shape)
        self.dtype = "torch.bfloat16"
        self.device = _Device()
        self._storage_bytes = (
            storage_bytes
            if storage_bytes is not None
            else self.numel() * self.element_size()
        )

    def numel(self) -> int:
        value = 1
        for dimension in self.shape:
            value *= dimension
        return value

    @staticmethod
    def element_size() -> int:
        return 2

    def untyped_storage(self):
        return _Storage(self._storage_bytes)


class _Manager:
    global_num_experts = 512
    local_num_experts = 32

    @staticmethod
    def map_global_to_local(expert_id: int) -> int:
        return expert_id if expert_id < 32 else -1


class _Owner:
    round16_logical_expert_count = 512
    round16_physical_expert_count = 32
    round16_resident_logical_ids = tuple(range(32))
    round16_logical_to_physical = tuple(range(32)) + (-1,) * 480
    round16_source_bound_api_identity = (
        "round16-b:source-bound-q32-routed-experts:offline-v1"
    )
    runtime_runner_binding_verified = False
    local_num_experts = 32
    global_num_experts = 512
    w13_weight = _Tensor((32, 1280, 2560))
    w2_weight = _Tensor((32, 2560, 640))
    expert_map_manager = _Manager()
    moe_config = SimpleNamespace(num_local_experts=32)
    round16_upstream_moe_config = SimpleNamespace(num_local_experts=512)
    quant_method = SimpleNamespace(
        __class__=SimpleNamespace(__name__="SourceBoundQ32QuantMethodAdapter")
    )


class _TorchNpu:
    bfloat16 = "torch.bfloat16"


class Q32PreallocationProbeTests(unittest.TestCase):
    def test_validates_real_sized_q32_owner_without_promoting_live_runtime(self):
        probe = _probe()
        snapshot = SimpleNamespace(
            resident_logical_ids=tuple(range(32)),
            logical_to_physical=tuple(range(32)) + (-1,) * 480,
            w13_shape=(32, 2560, 1280),
            w2_shape=(32, 640, 2560),
        )
        cleanup_calls = []
        binding = probe.Q32PreallocationBinding(
            owner=_Owner(),
            torch_module=_TorchNpu(),
            memory_before_bytes=1024,
            memory_after_bytes=314_573_824,
            peak_memory_bytes=314_573_824,
            construction_method="UnquantizedFusedMoEMethod",
            cleanup=lambda: cleanup_calls.append(True),
        )

        receipt = probe.validate_and_release(binding, snapshot, layer_id=0)

        self.assertEqual(
            receipt["status"], "REAL_IMAGE_Q32_CONSTRUCTOR_VERIFIED"
        )
        self.assertEqual(receipt["stage"], "V1")
        self.assertEqual(receipt["logical_experts"], 512)
        self.assertEqual(receipt["physical_npu_slots"], 32)
        self.assertEqual(receipt["payload_bytes"], 314_572_800)
        self.assertEqual(receipt["w13_raw_shape"], [32, 1280, 2560])
        self.assertEqual(receipt["w2_raw_shape"], [32, 2560, 640])
        self.assertFalse(receipt["runtime_runner_binding_verified"])
        self.assertEqual(cleanup_calls, [True])

    def test_rejects_hidden_full512_storage_and_still_releases(self):
        probe = _probe()
        snapshot = SimpleNamespace(
            resident_logical_ids=tuple(range(32)),
            logical_to_physical=tuple(range(32)) + (-1,) * 480,
            w13_shape=(32, 2560, 1280),
            w2_shape=(32, 640, 2560),
        )
        owner = _Owner()
        owner.w13_weight = _Tensor(
            (32, 1280, 2560),
            storage_bytes=512 * 1280 * 2560 * 2,
        )
        cleanup_calls = []
        binding = probe.Q32PreallocationBinding(
            owner=owner,
            torch_module=_TorchNpu(),
            memory_before_bytes=0,
            memory_after_bytes=0,
            peak_memory_bytes=0,
            construction_method="UnquantizedFusedMoEMethod",
            cleanup=lambda: cleanup_calls.append(True),
        )

        with self.assertRaisesRegex(ValueError, "larger backing storage"):
            probe.validate_and_release(binding, snapshot, layer_id=0)
        self.assertEqual(cleanup_calls, [True])

    def test_factory_identity_and_explicit_device_enable_are_required(self):
        probe = _probe()
        with self.assertRaisesRegex(RuntimeError, "ROUND16_B_ENABLE_Q32_V1"):
            probe.require_device_enable({})

        def factory(_context):
            raise AssertionError("unverified factory must not run")

        with self.assertRaisesRegex(ValueError, "factory API identity"):
            probe.validate_factory(factory)

    def test_cross_module_protocol_uses_api_identity_not_python_class_identity(self):
        probe = _probe()
        snapshot = SimpleNamespace(
            resident_logical_ids=tuple(range(32)),
            logical_to_physical=tuple(range(32)) + (-1,) * 480,
            w13_shape=(32, 2560, 1280),
            w2_shape=(32, 640, 2560),
        )
        cleanup_calls = []
        foreign_binding = SimpleNamespace(
            round16_api_identity=probe.Q32_V1_BINDING_API,
            owner=_Owner(),
            torch_module=_TorchNpu(),
            memory_before_bytes=0,
            memory_after_bytes=314_572_800,
            peak_memory_bytes=314_572_800,
            construction_method="UnquantizedFusedMoEMethod",
            cleanup=lambda: cleanup_calls.append(True),
        )
        receipt = probe.validate_and_release(foreign_binding, snapshot, layer_id=0)
        self.assertEqual(receipt["status"], "REAL_IMAGE_Q32_CONSTRUCTOR_VERIFIED")
        self.assertEqual(cleanup_calls, [True])

    def test_validate_only_writes_device_pending_receipt(self):
        probe = _probe()
        repository = Path(__file__).resolve().parents[2]
        manifest = repository / (
            "artifacts/round16-b/device-overlay/"
            "P1_REMOTE_SNAPSHOT_PREFLIGHT_MANIFEST.json"
        )
        preflight = importlib.import_module("round16.b.native_source_preflight")
        token_payload = {
            "api_checks": [item.name for item in preflight.DEFAULT_SOURCE_EXPECTATIONS]
            + ["model_config"],
            "observed_sha256": {
                **{
                    item.name: item.sha256
                    for item in preflight.DEFAULT_SOURCE_EXPECTATIONS
                },
                "model_config": preflight.DEFAULT_MODEL_CONFIG_EXPECTATION.sha256,
            },
            "passed": True,
            "status": "PASS_EXACT_SOURCE_AND_API",
        }
        with tempfile.TemporaryDirectory() as directory:
            token = Path(directory) / "token.json"
            output = Path(directory) / "receipt.json"
            token.write_text(json.dumps(token_payload))
            rc = probe.main(
                [
                    "--validate-only",
                    "--source-token",
                    str(token),
                    "--snapshot-manifest",
                    str(manifest),
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(rc, 0)
            receipt = json.loads(output.read_text())
            self.assertEqual(receipt["status"], "READY_FOR_Q32_V1_DEVICE_PROBE")
            self.assertFalse(receipt["device_executed"])


if __name__ == "__main__":
    unittest.main()
