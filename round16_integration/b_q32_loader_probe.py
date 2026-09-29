"""Bounded real-image V2 probe for selective Q32 load and Ascend post-load.

V2 consumes a successful V1 constructor receipt from the same frozen source
and S0 snapshot.  It proves that packed logical-512 checkpoint tensors are
sliced on CPU, loaded through native parameter callbacks into physical Q32,
then transformed by the exact Ascend BF16 post-load method without a hidden
full-expert NPU payload.  Hybrid execution remains a separate V3 gate.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from round16.b.native_source_preflight import validate_exact_source_preflight_token
from round16.b.resident_packed_loader import (
    PACKED_DOWN_SHAPE,
    PACKED_GATE_UP_SHAPE,
    ProjectionLoad,
    SELECTED_DOWN_SHAPE,
    SELECTED_GATE_UP_SHAPE,
    validate_projection_coverage,
)
from round16_integration.b_q32_preallocation_probe import (
    BF16_ELEMENT_BYTES,
    POSTLOAD_W13_SHAPE,
    POSTLOAD_W2_SHAPE,
    Q32_PAYLOAD_BYTES,
    _storage_nbytes,
)
from round16_integration.p1_allgather_q32_probe import (
    P1Snapshot,
    load_snapshot_manifest,
    load_source_preflight_token,
)


Q32_V2_PROBE_API = "round16-integration:q32-loader-probe:v1"
Q32_V2_FACTORY_API = "round16-integration:q32-loader-factory:v1"
Q32_V2_CONTEXT_API = "round16-integration:q32-loader-context:v1"
Q32_V2_BINDING_API = "round16-integration:q32-loader-binding:v1"
ASCEND_NZ_FORMAT = 29
FULL512_PAYLOAD_BYTES = 512 * (1280 * 2560 + 2560 * 640) * BF16_ELEMENT_BYTES
V2_PEAK_LIMIT_BYTES = 2 * Q32_PAYLOAD_BYTES + 256 * 1024 * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_v1_receipt(path: Path) -> tuple[dict[str, Any], str]:
    payload = json.loads(path.read_text())
    required = {
        "status": "REAL_IMAGE_Q32_CONSTRUCTOR_VERIFIED",
        "stage": "V1",
        "device_executed": True,
        "logical_experts": 512,
        "physical_npu_slots": 32,
        "payload_bytes": Q32_PAYLOAD_BYTES,
        "construction_method": "UnquantizedFusedMoEMethod",
    }
    for name, expected in required.items():
        if payload.get(name) != expected:
            raise ValueError(f"V2 rejected V1 receipt field {name}")
    if payload.get("loader_and_postload") != "NOT_RUN_V2":
        raise ValueError("V2 requires the original V1-only receipt")
    return payload, _sha256(path)


@dataclass(frozen=True)
class Q32V2FactoryContext:
    round16_api_identity = Q32_V2_CONTEXT_API

    source_token: Any
    snapshot: P1Snapshot
    layer_id: int
    checkpoint_root: Path
    v1_receipt_sha256: str


@dataclass(frozen=True)
class Q32LoaderBinding:
    round16_api_identity = Q32_V2_BINDING_API

    owner: Any
    torch_module: Any
    coverage: tuple[ProjectionLoad, ...]
    construction_method: str
    execution_method: str
    checkpoint_keys: Mapping[str, str]
    checkpoint_files: Mapping[str, Mapping[str, Any]]
    checkpoint_shapes: Mapping[str, Sequence[int]]
    checkpoint_dtypes: Mapping[str, str]
    source_row_reads: tuple[Mapping[str, Any], ...]
    selected_cpu_bytes: int
    load_mode: str
    postload_order: str
    exact_value_checks: Mapping[str, bool]
    w13_npu_format: int
    w2_npu_format: int
    worker_internal_format_assignment_issued: bool
    allocator_before_bytes: int
    allocator_after_load_bytes: int
    allocator_after_postload_bytes: int
    allocator_peak_bytes: int
    cleanup: Callable[[], Any]

    def __post_init__(self) -> None:
        if not callable(self.cleanup):
            raise TypeError("V2 binding cleanup must be callable")
        for name in (
            "selected_cpu_bytes",
            "allocator_before_bytes",
            "allocator_after_load_bytes",
            "allocator_after_postload_bytes",
            "allocator_peak_bytes",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")


def require_device_enable(environment: Mapping[str, str]) -> None:
    if environment.get("ROUND16_B_ENABLE_Q32_V2") != "1":
        raise RuntimeError("ROUND16_B_ENABLE_Q32_V2=1 is required for V2 device execution")


def validate_factory(factory: Any) -> None:
    if not callable(factory):
        raise TypeError("V2 binding factory must be callable")
    if getattr(factory, "round16_api_identity", None) != Q32_V2_FACTORY_API:
        raise ValueError("V2 binding factory API identity is not verified")


def _load_factory(specification: str) -> Callable[[Q32V2FactoryContext], Any]:
    if type(specification) is not str or specification.count(":") != 1:
        raise ValueError("V2 binding factory must use module:callable syntax")
    module_name, attribute = specification.split(":", 1)
    factory = getattr(importlib.import_module(module_name), attribute)
    validate_factory(factory)
    return factory


def _tensor_metadata(value: Any, name: str, expected_shape: tuple[int, ...]) -> dict[str, Any]:
    shape = tuple(getattr(value, "shape", ()))
    if shape != expected_shape:
        raise ValueError(f"{name} post-load shape differs from S0")
    if str(getattr(value, "dtype", "")).lower() not in (
        "torch.bfloat16",
        "bfloat16",
        "bf16",
    ):
        raise ValueError(f"{name} post-load dtype is not BF16")
    device = getattr(value, "device", None)
    if str(getattr(device, "type", "")).lower() != "npu":
        raise ValueError(f"{name} post-load device is not NPU")
    tensor_bytes = int(value.numel()) * int(value.element_size())
    storage_bytes = _storage_nbytes(value, name)
    if storage_bytes >= FULL512_PAYLOAD_BYTES:
        raise ValueError(f"{name} retains a hidden full512 expert backing storage")
    if storage_bytes > tensor_bytes + 128 * 1024 * 1024:
        raise ValueError(f"{name} post-load storage exceeds bounded Q32 padding")
    return {
        "shape": list(shape),
        "dtype": str(value.dtype),
        "device": str(device),
        "tensor_bytes": tensor_bytes,
        "storage_bytes": storage_bytes,
    }


def validate_and_release(
    binding: Q32LoaderBinding,
    snapshot: P1Snapshot,
    *,
    layer_id: int,
    v1_receipt_sha256: str,
) -> dict[str, Any]:
    if getattr(binding, "round16_api_identity", None) != Q32_V2_BINDING_API:
        raise TypeError("V2 factory returned an unverified binding contract")
    try:
        validate_projection_coverage(binding.coverage, binding.owner.round16_selective_plan)
        if tuple(binding.owner.round16_resident_logical_ids) != tuple(
            snapshot.resident_logical_ids
        ):
            raise ValueError("V2 owner resident map differs from S0")
        if binding.construction_method != "UnquantizedFusedMoEMethod":
            raise ValueError("V2 construction did not use upstream BF16 method")
        if binding.execution_method != "AscendUnquantizedFusedMoEMethod":
            raise ValueError("V2 post-load did not use exact Ascend BF16 method")
        if binding.postload_order != "UPSTREAM_CREATE_LOAD_THEN_ASCEND_POSTLOAD":
            raise ValueError("V2 BF16 initialization order is invalid")
        if binding.load_mode != "SAFETENSORS_CPU_ROW_SLICE_NATIVE_PARAMETER_LOADER":
            raise ValueError("V2 checkpoint load mode is not resident-selective")
        if tuple(binding.checkpoint_shapes.get("gate_up", ())) != PACKED_GATE_UP_SHAPE:
            raise ValueError("V2 packed gate/up source shape is invalid")
        if tuple(binding.checkpoint_shapes.get("down", ())) != PACKED_DOWN_SHAPE:
            raise ValueError("V2 packed down source shape is invalid")
        if {value.upper() for value in binding.checkpoint_dtypes.values()} != {"BF16"}:
            raise ValueError("V2 packed checkpoint dtype is not BF16")
        if binding.selected_cpu_bytes != Q32_PAYLOAD_BYTES:
            raise ValueError("V2 selected CPU payload is not exactly Q32")
        reads = tuple(binding.source_row_reads)
        if len(reads) != 64:
            raise ValueError("V2 must perform exactly 32 gate/up and 32 down row reads")
        if any(read.get("logical_expert_id") not in snapshot.resident_logical_ids for read in reads):
            raise ValueError("V2 read a nonresident checkpoint expert row")
        if any(read.get("materialized_leading_experts") != 1 for read in reads):
            raise ValueError("V2 materialized more than one source expert per row read")
        if set(binding.exact_value_checks) != {
            "gate_full_q32_exact",
            "up_full_q32_exact",
            "down_full_q32_exact",
        } or not all(binding.exact_value_checks.values()):
            raise ValueError("V2 post-load values differ from selected checkpoint rows")
        w13 = _tensor_metadata(binding.owner.w13_weight, "w13_weight", POSTLOAD_W13_SHAPE)
        w2 = _tensor_metadata(binding.owner.w2_weight, "w2_weight", POSTLOAD_W2_SHAPE)
        if binding.w13_npu_format != ASCEND_NZ_FORMAT or binding.w2_npu_format != ASCEND_NZ_FORMAT:
            raise ValueError("V2 post-load expert weights are not Ascend NZ29")
        if binding.worker_internal_format_assignment_issued is not True:
            raise ValueError("V2 worker internal-format assignment was not issued")
        peak_delta = max(0, binding.allocator_peak_bytes - binding.allocator_before_bytes)
        if peak_delta > V2_PEAK_LIMIT_BYTES:
            raise ValueError("V2 allocator peak exceeds two Q32 payloads plus bounded workspace")
        return {
            "schema": 1,
            "status": "REAL_Q32_LOADER_AND_LAYOUT_VERIFIED",
            "stage": "V2",
            "device_executed": True,
            "layer_id": layer_id,
            "logical_experts": 512,
            "physical_npu_slots": 32,
            "resident_logical_ids": list(snapshot.resident_logical_ids),
            "v1_receipt_sha256": v1_receipt_sha256,
            "construction_method": binding.construction_method,
            "execution_method": binding.execution_method,
            "postload_order": binding.postload_order,
            "load_mode": binding.load_mode,
            "checkpoint_keys": dict(binding.checkpoint_keys),
            "checkpoint_files": dict(binding.checkpoint_files),
            "checkpoint_shapes": {
                name: list(shape) for name, shape in binding.checkpoint_shapes.items()
            },
            "checkpoint_dtypes": dict(binding.checkpoint_dtypes),
            "selected_cpu_bytes": binding.selected_cpu_bytes,
            "source_row_read_count": len(reads),
            "projection_coverage_count": len(binding.coverage),
            "projection_coverage": [
                {
                    "logical_expert_id": item.logical_expert_id,
                    "physical_slot": item.physical_slot,
                    "projection": item.projection,
                }
                for item in binding.coverage
            ],
            "exact_value_checks": dict(binding.exact_value_checks),
            "w13_postload": w13,
            "w2_postload": w2,
            "w13_npu_format": binding.w13_npu_format,
            "w2_npu_format": binding.w2_npu_format,
            "worker_internal_format_assignment_issued": (
                binding.worker_internal_format_assignment_issued
            ),
            "allocator_before_bytes": binding.allocator_before_bytes,
            "allocator_after_load_bytes": binding.allocator_after_load_bytes,
            "allocator_after_postload_bytes": binding.allocator_after_postload_bytes,
            "allocator_peak_bytes": binding.allocator_peak_bytes,
            "allocator_peak_delta_bytes": peak_delta,
            "hidden_full512_npu_payload": False,
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
    parser.add_argument("--v1-receipt", required=True, type=Path)
    parser.add_argument("--checkpoint-root", required=True, type=Path)
    parser.add_argument("--binding-factory")
    parser.add_argument("--layer-id", type=int, default=0)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    parsed = _parser().parse_args(arguments)
    source_token = load_source_preflight_token(parsed.source_token)
    if not validate_exact_source_preflight_token(source_token):
        raise ValueError("V2 exact source token is rejected")
    snapshot = load_snapshot_manifest(parsed.snapshot_manifest)
    _, v1_sha = load_v1_receipt(parsed.v1_receipt)
    if not (0 <= parsed.layer_id < 48):
        raise ValueError("V2 layer id must be in [0,48)")
    if not parsed.checkpoint_root.is_dir():
        raise ValueError("V2 checkpoint root does not exist")
    context = Q32V2FactoryContext(
        source_token,
        snapshot,
        parsed.layer_id,
        parsed.checkpoint_root,
        v1_sha,
    )
    if parsed.validate_only:
        _write_json(
            parsed.output,
            {
                "schema": 1,
                "status": "READY_FOR_Q32_V2_DEVICE_PROBE",
                "device_executed": False,
                "v1_receipt_sha256": v1_sha,
                "logical_experts": 512,
                "physical_npu_slots": 32,
            },
        )
        return 0
    require_device_enable(os.environ)
    if not parsed.binding_factory:
        raise ValueError("V2 device execution requires --binding-factory")
    binding = _load_factory(parsed.binding_factory)(context)
    _write_json(
        parsed.output,
        validate_and_release(
            binding,
            snapshot,
            layer_id=parsed.layer_id,
            v1_receipt_sha256=v1_sha,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "Q32LoaderBinding",
    "Q32V2FactoryContext",
    "Q32_V2_BINDING_API",
    "Q32_V2_CONTEXT_API",
    "Q32_V2_FACTORY_API",
    "Q32_V2_PROBE_API",
    "load_v1_receipt",
    "require_device_enable",
    "validate_and_release",
)
