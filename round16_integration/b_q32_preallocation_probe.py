"""Bounded real-image V1 probe for physical-Q32 expert preallocation.

The module itself imports no torch, vLLM, vLLM-Ascend, checkpoint, or device
runtime.  Static source and S0 snapshot identities are validated before an
explicit factory is imported.  V1 proves only constructor/storage capacity;
it deliberately leaves checkpoint loading, post-load Ascend layout, Hybrid
execution, and the full model unqualified.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from round16.b.native_source_preflight import (
    validate_exact_source_preflight_token,
)
from round16.b.source_bound_q32_adapter import (
    SOURCE_BOUND_Q32_ROUTED_EXPERTS_API,
)
from round16_integration.p1_allgather_q32_probe import (
    P1Snapshot,
    load_snapshot_manifest,
    load_source_preflight_token,
)


Q32_V1_PROBE_API = "round16-integration:q32-preallocation-probe:v1"
Q32_V1_FACTORY_API = "round16-integration:q32-preallocation-factory:v1"
Q32_V1_CONTEXT_API = "round16-integration:q32-preallocation-context:v1"
Q32_V1_BINDING_API = "round16-integration:q32-preallocation-binding:v1"
RAW_W13_SHAPE = (32, 1280, 2560)
RAW_W2_SHAPE = (32, 2560, 640)
POSTLOAD_W13_SHAPE = (32, 2560, 1280)
POSTLOAD_W2_SHAPE = (32, 640, 2560)
BF16_ELEMENT_BYTES = 2
Q32_PAYLOAD_BYTES = 314_572_800
PEAK_OVERHEAD_LIMIT_BYTES = 128 * 1024 * 1024


@dataclass(frozen=True)
class Q32V1FactoryContext:
    round16_api_identity = Q32_V1_CONTEXT_API

    source_token: Any
    snapshot: P1Snapshot
    layer_id: int


@dataclass(frozen=True)
class Q32PreallocationBinding:
    round16_api_identity = Q32_V1_BINDING_API

    owner: Any
    torch_module: Any
    memory_before_bytes: int
    memory_after_bytes: int
    peak_memory_bytes: int
    construction_method: str
    cleanup: Callable[[], Any]

    def __post_init__(self) -> None:
        for name in (
            "memory_before_bytes",
            "memory_after_bytes",
            "peak_memory_bytes",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if not callable(self.cleanup):
            raise TypeError("V1 binding cleanup must be callable")
        if type(self.construction_method) is not str or not self.construction_method:
            raise ValueError("V1 construction method identity is required")


def require_device_enable(environment: Mapping[str, str]) -> None:
    if environment.get("ROUND16_B_ENABLE_Q32_V1") != "1":
        raise RuntimeError(
            "ROUND16_B_ENABLE_Q32_V1=1 is required for V1 device execution"
        )


def validate_factory(factory: Any) -> None:
    if not callable(factory):
        raise TypeError("V1 binding factory must be callable")
    if getattr(factory, "round16_api_identity", None) != Q32_V1_FACTORY_API:
        raise ValueError("V1 binding factory API identity is not verified")


def _load_factory(specification: str) -> Callable[[Q32V1FactoryContext], Any]:
    if type(specification) is not str or specification.count(":") != 1:
        raise ValueError("V1 binding factory must use module:callable syntax")
    module_name, attribute = specification.split(":", 1)
    if not module_name or not attribute:
        raise ValueError("V1 binding factory must use module:callable syntax")
    factory = getattr(importlib.import_module(module_name), attribute)
    validate_factory(factory)
    return factory


def _shape(value: Any, name: str) -> tuple[int, ...]:
    try:
        result = tuple(value.shape)
    except (AttributeError, TypeError) as error:
        raise TypeError(f"{name} does not expose a tensor shape") from error
    if not result or any(type(item) is not int or item <= 0 for item in result):
        raise ValueError(f"{name} has invalid tensor shape")
    return result


def _storage_nbytes(value: Any, name: str) -> int:
    storage_factory = getattr(value, "untyped_storage", None)
    if not callable(storage_factory):
        raise TypeError(f"{name} does not expose untyped storage")
    storage = storage_factory()
    nbytes = getattr(storage, "nbytes", None)
    result = nbytes() if callable(nbytes) else nbytes
    if type(result) is not int or result <= 0:
        raise ValueError(f"{name} storage byte count is invalid")
    return result


def _payload_metadata(value: Any, name: str, expected_shape: tuple[int, ...]) -> dict:
    shape = _shape(value, name)
    if shape != expected_shape:
        raise ValueError(f"{name} raw constructor shape differs from physical Q32")
    dtype = str(getattr(value, "dtype", "")).lower()
    if dtype not in ("torch.bfloat16", "bfloat16", "bf16"):
        raise ValueError(f"{name} must use BF16")
    device = getattr(value, "device", None)
    if str(getattr(device, "type", "")).lower() != "npu":
        raise ValueError(f"{name} must be allocated on NPU")
    try:
        numel = value.numel()
        element_size = value.element_size()
    except (AttributeError, TypeError) as error:
        raise TypeError(f"{name} does not expose allocation size") from error
    if type(numel) is not int or type(element_size) is not int:
        raise ValueError(f"{name} allocation size is invalid")
    tensor_bytes = numel * element_size
    storage_bytes = _storage_nbytes(value, name)
    if element_size != BF16_ELEMENT_BYTES or storage_bytes != tensor_bytes:
        raise ValueError(
            f"{name} is backed by larger backing storage; a hidden full-expert "
            "payload or view is forbidden"
        )
    return {
        "shape": list(shape),
        "dtype": dtype,
        "device": str(device),
        "tensor_bytes": tensor_bytes,
        "storage_bytes": storage_bytes,
    }


def _validate_mapping(owner: Any, snapshot: P1Snapshot) -> None:
    if tuple(getattr(owner, "round16_resident_logical_ids", ())) != tuple(
        snapshot.resident_logical_ids
    ):
        raise ValueError("V1 owner resident identities differ from S0")
    if tuple(getattr(owner, "round16_logical_to_physical", ())) != tuple(
        snapshot.logical_to_physical
    ):
        raise ValueError("V1 owner logical-to-physical map differs from S0")
    if getattr(owner, "round16_logical_expert_count", None) != 512:
        raise ValueError("V1 owner must preserve 512 logical experts")
    if getattr(owner, "round16_physical_expert_count", None) != 32:
        raise ValueError("V1 owner must allocate exactly 32 physical experts")
    if getattr(owner, "global_num_experts", None) != 512:
        raise ValueError("V1 owner global expert count must remain 512")
    if getattr(owner, "local_num_experts", None) != 32:
        raise ValueError("V1 owner physical local count must be 32")
    manager = getattr(owner, "expert_map_manager", None)
    if (
        getattr(manager, "global_num_experts", None) != 512
        or getattr(manager, "local_num_experts", None) != 32
    ):
        raise ValueError("V1 physical manager dimensions must be logical512/Q32")
    mapper = getattr(manager, "map_global_to_local", None)
    if not callable(mapper):
        raise TypeError("V1 physical manager has no mapper")
    for logical_id, expected in enumerate(snapshot.logical_to_physical):
        if mapper(logical_id) != expected:
            raise ValueError(
                f"V1 physical manager mapping differs at logical expert {logical_id}"
            )


def validate_and_release(
    binding: Q32PreallocationBinding,
    snapshot: P1Snapshot,
    *,
    layer_id: int,
) -> dict[str, Any]:
    """Validate one real owner and always invoke its bounded release hook."""

    if getattr(binding, "round16_api_identity", None) != Q32_V1_BINDING_API:
        raise TypeError("V1 factory returned an unverified binding contract")
    if not isinstance(snapshot, P1Snapshot) and not all(
        hasattr(snapshot, name)
        for name in (
            "resident_logical_ids",
            "logical_to_physical",
            "w13_shape",
            "w2_shape",
        )
    ):
        raise TypeError("V1 snapshot contract is invalid")
    if type(layer_id) is not int or not (0 <= layer_id < 48):
        raise ValueError("V1 layer id must be in [0,48)")
    owner = binding.owner
    try:
        _validate_mapping(owner, snapshot)
        if (
            getattr(owner, "round16_source_bound_api_identity", None)
            != SOURCE_BOUND_Q32_ROUTED_EXPERTS_API
        ):
            raise ValueError("V1 owner source-bound API identity mismatch")
        upstream_config = getattr(owner, "round16_upstream_moe_config", None)
        physical_config = getattr(owner, "moe_config", None)
        if upstream_config is physical_config:
            raise ValueError("V1 owner mutated the shared logical config")
        if getattr(upstream_config, "num_local_experts", None) != 512:
            raise ValueError("V1 upstream semantic config must remain local512")
        if getattr(physical_config, "num_local_experts", None) != 32:
            raise ValueError("V1 physical config must be Q32 before allocation")
        if binding.construction_method != "UnquantizedFusedMoEMethod":
            raise ValueError(
                "V1 BF16 weights must be created by upstream "
                "UnquantizedFusedMoEMethod before Ascend execution takeover"
            )
        w13 = _payload_metadata(owner.w13_weight, "w13_weight", RAW_W13_SHAPE)
        w2 = _payload_metadata(owner.w2_weight, "w2_weight", RAW_W2_SHAPE)
        payload_bytes = w13["storage_bytes"] + w2["storage_bytes"]
        if payload_bytes != Q32_PAYLOAD_BYTES:
            raise ValueError("V1 physical Q32 payload byte count is incorrect")
        if tuple(snapshot.w13_shape) != POSTLOAD_W13_SHAPE or tuple(
            snapshot.w2_shape
        ) != POSTLOAD_W2_SHAPE:
            raise ValueError("S0 post-load layout does not match the frozen Q32 receipt")
        peak_delta = max(
            0, binding.peak_memory_bytes - binding.memory_before_bytes
        )
        if peak_delta > Q32_PAYLOAD_BYTES + PEAK_OVERHEAD_LIMIT_BYTES:
            raise ValueError(
                "V1 allocator peak exceeds one Q32 payload plus bounded overhead; "
                "a full-expert temporary allocation is possible"
            )
        return {
            "schema": 1,
            "status": "REAL_IMAGE_Q32_CONSTRUCTOR_VERIFIED",
            "stage": "V1",
            "device_executed": True,
            "layer_id": layer_id,
            "logical_experts": 512,
            "physical_npu_slots": 32,
            "construction_method": binding.construction_method,
            "bf16_initialization_order": "UPSTREAM_CREATE_THEN_ASCEND_POSTLOAD_PENDING",
            "payload_bytes": payload_bytes,
            "w13_raw_shape": w13["shape"],
            "w2_raw_shape": w2["shape"],
            "w13_storage_bytes": w13["storage_bytes"],
            "w2_storage_bytes": w2["storage_bytes"],
            "allocator_before_bytes": binding.memory_before_bytes,
            "allocator_after_bytes": binding.memory_after_bytes,
            "allocator_peak_bytes": binding.peak_memory_bytes,
            "allocator_peak_delta_bytes": peak_delta,
            "runtime_runner_binding_verified": bool(
                getattr(owner, "runtime_runner_binding_verified", False)
            ),
            "loader_and_postload": "NOT_RUN_V2",
            "hybrid_execution": "NOT_RUN_V3",
            "all_layer_residency": "NOT_RUN_V4",
            "full_model": "NOT_RUN_V5",
        }
    finally:
        binding.cleanup()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--enable-device", action="store_true")
    parser.add_argument("--source-token", required=True, type=Path)
    parser.add_argument("--snapshot-manifest", required=True, type=Path)
    parser.add_argument("--binding-factory")
    parser.add_argument("--layer-id", type=int, default=0)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    parsed = _parser().parse_args(arguments)
    source_token = load_source_preflight_token(parsed.source_token)
    if not validate_exact_source_preflight_token(source_token):
        raise ValueError("V1 exact source token is rejected")
    snapshot = load_snapshot_manifest(parsed.snapshot_manifest)
    context = Q32V1FactoryContext(source_token, snapshot, parsed.layer_id)
    if parsed.validate_only:
        _write_json(
            parsed.output,
            {
                "schema": 1,
                "status": "READY_FOR_Q32_V1_DEVICE_PROBE",
                "device_executed": False,
                "required_factory_api": Q32_V1_FACTORY_API,
                "logical_experts": 512,
                "physical_npu_slots": 32,
            },
        )
        return 0
    require_device_enable(os.environ)
    if not parsed.binding_factory:
        raise ValueError("V1 device execution requires --binding-factory")
    factory = _load_factory(parsed.binding_factory)
    binding = factory(context)
    _write_json(
        parsed.output,
        validate_and_release(binding, snapshot, layer_id=parsed.layer_id),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "Q32PreallocationBinding",
    "Q32V1FactoryContext",
    "Q32_V1_FACTORY_API",
    "Q32_V1_PROBE_API",
    "require_device_enable",
    "validate_and_release",
    "validate_factory",
)
