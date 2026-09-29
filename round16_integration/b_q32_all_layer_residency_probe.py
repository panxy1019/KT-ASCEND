"""Round16-B V4 probe for simultaneous 48-layer physical-Q32 residency.

V4 deliberately constructs only the real expert owners.  It does not read the
checkpoint, create CPUInfer, or construct the model backbone.  The purpose is
to prove that all 48 owners can coexist while each retains the frozen
logical-512/physical-32 mapping and owns only its two Q32 BF16 payloads.
"""

from __future__ import annotations

import argparse
import gc
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
from round16_integration.b_q32_preallocation_probe import (
    BF16_ELEMENT_BYTES,
    Q32_PAYLOAD_BYTES,
    Q32_V1_BINDING_API,
    Q32V1FactoryContext,
    RAW_W13_SHAPE,
    RAW_W2_SHAPE,
    _load_factory,
)
from round16_integration.p1_allgather_q32_probe import (
    P1Snapshot,
    load_snapshot_manifest,
    load_source_preflight_token,
)


Q32_V4_PROBE_API = "round16-integration:q32-all-layer-residency-probe:v1"
EXPECTED_LAYER_COUNT = 48
AGGREGATE_PAYLOAD_BYTES = EXPECTED_LAYER_COUNT * Q32_PAYLOAD_BYTES
PER_LAYER_OVERHEAD_LIMIT_BYTES = 8 * 1024 * 1024
AGGREGATE_OVERHEAD_LIMIT_BYTES = 256 * 1024 * 1024


def require_device_enable(environment: Mapping[str, str]) -> None:
    if environment.get("ROUND16_B_ENABLE_Q32_V4") != "1":
        raise RuntimeError(
            "ROUND16_B_ENABLE_Q32_V4=1 is required for V4 device execution"
        )


def _shape(value: Any, name: str) -> tuple[int, ...]:
    try:
        result = tuple(value.shape)
    except (AttributeError, TypeError) as error:
        raise TypeError(f"{name} does not expose a tensor shape") from error
    if not result or any(type(item) is not int or item <= 0 for item in result):
        raise ValueError(f"{name} has an invalid shape")
    return result


def _storage_metadata(
    value: Any, name: str, expected_shape: tuple[int, ...]
) -> dict[str, Any]:
    shape = _shape(value, name)
    if shape != expected_shape:
        raise ValueError(f"{name} is not a raw physical-Q32 tensor")
    if str(getattr(value, "dtype", "")).lower() not in (
        "torch.bfloat16",
        "bfloat16",
        "bf16",
    ):
        raise ValueError(f"{name} must use BF16")
    if str(getattr(getattr(value, "device", None), "type", "")).lower() != "npu":
        raise ValueError(f"{name} must be on NPU")
    storage_factory = getattr(value, "untyped_storage", None)
    if not callable(storage_factory):
        raise TypeError(f"{name} does not expose untyped storage")
    storage = storage_factory()
    nbytes_value = getattr(storage, "nbytes", None)
    storage_bytes = nbytes_value() if callable(nbytes_value) else nbytes_value
    if type(storage_bytes) is not int or storage_bytes <= 0:
        raise ValueError(f"{name} storage byte count is invalid")
    expected_bytes = 1
    for dimension in shape:
        expected_bytes *= dimension
    expected_bytes *= BF16_ELEMENT_BYTES
    if storage_bytes != expected_bytes:
        raise ValueError(
            f"{name} is backed by larger storage; hidden full-expert payloads "
            "and views are forbidden"
        )
    pointer_factory = getattr(storage, "data_ptr", None)
    pointer = pointer_factory() if callable(pointer_factory) else None
    if type(pointer) is not int or pointer <= 0:
        tensor_pointer = getattr(value, "data_ptr", None)
        pointer = tensor_pointer() if callable(tensor_pointer) else None
    if type(pointer) is not int or pointer <= 0:
        raise ValueError(f"{name} storage pointer is unavailable")
    return {
        "shape": list(shape),
        "storage_bytes": storage_bytes,
        "storage_pointer": pointer,
    }


def _validate_mapping(owner: Any, snapshot: P1Snapshot, layer_id: int) -> None:
    if type(layer_id) is not int or not (0 <= layer_id < EXPECTED_LAYER_COUNT):
        raise ValueError("V4 layer ID must be in [0,48)")
    if getattr(owner, "round16_source_bound_api_identity", None) != (
        SOURCE_BOUND_Q32_ROUTED_EXPERTS_API
    ):
        raise ValueError("V4 owner source-bound identity differs")
    if getattr(owner, "round16_logical_expert_count", None) != 512:
        raise ValueError("V4 owner must preserve 512 logical experts")
    if getattr(owner, "round16_physical_expert_count", None) != 32:
        raise ValueError("V4 owner must allocate physical Q32")
    if tuple(getattr(owner, "round16_resident_logical_ids", ())) != tuple(
        snapshot.resident_logical_ids
    ):
        raise ValueError("V4 resident IDs differ from the frozen S0 map")
    if tuple(getattr(owner, "round16_logical_to_physical", ())) != tuple(
        snapshot.logical_to_physical
    ):
        raise ValueError("V4 logical-to-physical map differs from S0")
    manager = getattr(owner, "expert_map_manager", None)
    if (
        getattr(manager, "global_num_experts", None) != 512
        or getattr(manager, "local_num_experts", None) != 32
    ):
        raise ValueError("V4 manager must remain logical512/physical32")
    mapper = getattr(manager, "map_global_to_local", None)
    if not callable(mapper):
        raise TypeError("V4 owner manager has no mapper")
    for logical_id, expected in enumerate(snapshot.logical_to_physical):
        if mapper(logical_id) != expected:
            raise ValueError(f"V4 mapping differs at logical expert {logical_id}")


def validate_residency_set(
    bindings: Sequence[Any],
    snapshot: P1Snapshot,
    *,
    allocator_before_bytes: int,
    allocator_after_bytes: int,
    allocator_peak_bytes: int,
) -> dict[str, Any]:
    """Validate a complete simultaneously-live set without releasing it."""

    if len(bindings) != EXPECTED_LAYER_COUNT:
        raise ValueError("V4 requires exactly 48 simultaneously-live owners")
    pointers: set[int] = set()
    owner_ids: set[int] = set()
    records: list[dict[str, Any]] = []
    increments: list[int] = []
    prior_after = allocator_before_bytes
    for layer_id, binding in enumerate(bindings):
        if getattr(binding, "round16_api_identity", None) != Q32_V1_BINDING_API:
            raise TypeError("V4 factory returned an unverified V1 binding")
        owner = binding.owner
        if id(owner) in owner_ids:
            raise ValueError("V4 reused one owner object across multiple layers")
        owner_ids.add(id(owner))
        _validate_mapping(owner, snapshot, layer_id)
        w13 = _storage_metadata(owner.w13_weight, "w13_weight", RAW_W13_SHAPE)
        w2 = _storage_metadata(owner.w2_weight, "w2_weight", RAW_W2_SHAPE)
        if w13["storage_pointer"] == w2["storage_pointer"]:
            raise ValueError("V4 w13 and w2 share one storage")
        for pointer in (w13["storage_pointer"], w2["storage_pointer"]):
            if pointer in pointers:
                raise ValueError("V4 expert payload storage aliases another layer")
            pointers.add(pointer)
        if binding.construction_method != "UnquantizedFusedMoEMethod":
            raise ValueError("V4 owner did not use the qualified BF16 creator")
        increment = binding.memory_after_bytes - binding.memory_before_bytes
        if increment < Q32_PAYLOAD_BYTES or increment > (
            Q32_PAYLOAD_BYTES + PER_LAYER_OVERHEAD_LIMIT_BYTES
        ):
            raise ValueError("V4 per-layer allocation is outside the Q32 bound")
        if binding.memory_before_bytes < prior_after:
            raise ValueError("V4 allocator accounting moved backwards")
        prior_after = binding.memory_after_bytes
        increments.append(increment)
        records.append(
            {
                "layer_id": layer_id,
                "payload_bytes": w13["storage_bytes"] + w2["storage_bytes"],
                "allocator_increment_bytes": increment,
                "w13_storage_pointer": w13["storage_pointer"],
                "w2_storage_pointer": w2["storage_pointer"],
            }
        )
    allocated_delta = allocator_after_bytes - allocator_before_bytes
    peak_delta = allocator_peak_bytes - allocator_before_bytes
    for name, value in (
        ("allocator delta", allocated_delta),
        ("allocator peak delta", peak_delta),
    ):
        if value < AGGREGATE_PAYLOAD_BYTES or value > (
            AGGREGATE_PAYLOAD_BYTES + AGGREGATE_OVERHEAD_LIMIT_BYTES
        ):
            raise ValueError(f"V4 {name} is outside the all-layer Q32 bound")
    return {
        "schema": 1,
        "status": "ALL_LAYER_RESIDENCY_VERIFIED",
        "stage": "V4",
        "device_executed": True,
        "logical_experts_per_layer": 512,
        "physical_npu_slots_per_layer": 32,
        "layer_count": EXPECTED_LAYER_COUNT,
        "distinct_owner_count": len(owner_ids),
        "distinct_storage_count": len(pointers),
        "nominal_payload_bytes": AGGREGATE_PAYLOAD_BYTES,
        "allocator_before_bytes": allocator_before_bytes,
        "allocator_after_bytes": allocator_after_bytes,
        "allocator_delta_bytes": allocated_delta,
        "allocator_peak_bytes": allocator_peak_bytes,
        "allocator_peak_delta_bytes": peak_delta,
        "minimum_layer_increment_bytes": min(increments),
        "maximum_layer_increment_bytes": max(increments),
        "layers": records,
        "checkpoint_loading": "NOT_RUN_V4_CAPACITY_ONLY",
        "hybrid_execution": "INHERITED_FROM_SAME_IDENTITY_V3",
        "full_model": "NOT_RUN_V5",
    }


def run_device_probe(
    *, source_token: Any, snapshot: P1Snapshot, factory: Callable[[Any], Any]
) -> dict[str, Any]:
    require_device_enable(os.environ)
    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(0)
    torch.npu.synchronize()
    allocator_before = int(torch.npu.memory_allocated())
    torch.npu.reset_peak_memory_stats()
    bindings: list[Any] = []
    receipt: dict[str, Any] | None = None
    try:
        for layer_id in range(EXPECTED_LAYER_COUNT):
            context = Q32V1FactoryContext(source_token, snapshot, layer_id)
            bindings.append(factory(context))
        torch.npu.synchronize()
        allocator_after = int(torch.npu.memory_allocated())
        allocator_peak = int(torch.npu.max_memory_allocated())
        receipt = validate_residency_set(
            bindings,
            snapshot,
            allocator_before_bytes=allocator_before,
            allocator_after_bytes=allocator_after,
            allocator_peak_bytes=allocator_peak,
        )
    finally:
        for binding in reversed(bindings):
            try:
                binding.cleanup()
            except Exception:
                pass
        bindings.clear()
        gc.collect()
        torch.npu.synchronize()
        cleanup_allocated = int(torch.npu.memory_allocated())
        torch.npu.empty_cache()
    if receipt is None:
        raise RuntimeError("V4 did not produce a receipt")
    receipt["cleanup_allocator_bytes"] = cleanup_allocated
    receipt["cleanup_delta_from_entry_bytes"] = cleanup_allocated - allocator_before
    return receipt


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enable-device", action="store_true")
    parser.add_argument("--source-token", required=True, type=Path)
    parser.add_argument("--snapshot-manifest", required=True, type=Path)
    parser.add_argument(
        "--binding-factory",
        default=(
            "round16_integration.b_q32_preallocation_live_factory:"
            "create_q32_preallocation_binding"
        ),
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    parsed = _parser().parse_args(arguments)
    source_token = load_source_preflight_token(parsed.source_token)
    if not validate_exact_source_preflight_token(source_token):
        raise ValueError("V4 exact source token is rejected")
    snapshot = load_snapshot_manifest(parsed.snapshot_manifest)
    factory = _load_factory(parsed.binding_factory)
    if not parsed.enable_device:
        raise RuntimeError("V4 has no validate-only shortcut")
    result = run_device_probe(
        source_token=source_token, snapshot=snapshot, factory=factory
    )
    _write_json(parsed.output, result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
