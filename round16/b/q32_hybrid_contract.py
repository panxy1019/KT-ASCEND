"""Pure-Python contract for the bounded Round16-B V3 Hybrid MoE probe.

The device probe imports this module before importing torch, vLLM, torch_npu,
or the KT native extension.  It freezes the three routing partitions that are
required to prove the logical-512/physical-Q32 execution boundary and validates
the prior V2 and MXFP4-bank receipts without reading large weight payloads.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


Q32_V3_CONTRACT_API = "round16-b:q32-hybrid-contract:v1"
LOGICAL_EXPERTS = 512
PHYSICAL_NPU_EXPERTS = 32
TOP_K = 10
HIDDEN_SIZE = 2560
INTERMEDIATE_SIZE = 640


@dataclass(frozen=True)
class HybridRouteCase:
    name: str
    logical_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.name not in {"all_npu", "all_cpu", "mixed"}:
            raise ValueError("unknown V3 route case")
        if len(self.logical_ids) != TOP_K:
            raise ValueError("V3 route case must contain exactly Top-10 IDs")
        if any(type(value) is not int for value in self.logical_ids):
            raise TypeError("V3 route IDs must be exact integers")
        if any(not 0 <= value < LOGICAL_EXPERTS for value in self.logical_ids):
            raise ValueError("V3 route ID is outside the logical expert domain")
        if len(set(self.logical_ids)) != TOP_K:
            raise ValueError("V3 route case must use ten distinct experts")
        resident = tuple(value < PHYSICAL_NPU_EXPERTS for value in self.logical_ids)
        if self.name == "all_npu" and not all(resident):
            raise ValueError("all_npu contains a nonresident expert")
        if self.name == "all_cpu" and any(resident):
            raise ValueError("all_cpu contains a resident expert")
        if self.name == "mixed" and (all(resident) or not any(resident)):
            raise ValueError("mixed must contain both resident and CPU experts")


def frozen_route_cases() -> tuple[HybridRouteCase, ...]:
    """Return the immutable V3 all-NPU/all-CPU/mixed occurrence cases."""

    return (
        HybridRouteCase("all_npu", tuple(range(10))),
        HybridRouteCase("all_cpu", tuple(range(32, 42))),
        HybridRouteCase("mixed", (0, 32, 1, 33, 2, 34, 3, 35, 4, 36)),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_v2_receipt(path: Path) -> tuple[dict[str, Any], str]:
    payload = json.loads(path.read_text())
    required = {
        "schema": 1,
        "status": "REAL_Q32_LOADER_AND_LAYOUT_VERIFIED",
        "stage": "V2",
        "device_executed": True,
        "logical_experts": LOGICAL_EXPERTS,
        "physical_npu_slots": PHYSICAL_NPU_EXPERTS,
        "construction_method": "UnquantizedFusedMoEMethod",
        "execution_method": "AscendUnquantizedFusedMoEMethod",
        "postload_order": "UPSTREAM_CREATE_LOAD_THEN_ASCEND_POSTLOAD",
        "load_mode": "SAFETENSORS_CPU_ROW_SLICE_NATIVE_PARAMETER_LOADER",
        "w13_npu_format": 29,
        "w2_npu_format": 29,
        "hidden_full512_npu_payload": False,
        "hybrid_execution": "NOT_RUN_V3",
    }
    for name, expected in required.items():
        if payload.get(name) != expected:
            raise ValueError(f"V3 rejected V2 receipt field {name}")
    resident = tuple(payload.get("resident_logical_ids", ()))
    if resident != tuple(range(PHYSICAL_NPU_EXPERTS)):
        raise ValueError("V3 requires the frozen S0 layer0 resident IDs 0..31")
    if payload.get("projection_coverage_count") != 96:
        raise ValueError("V3 requires complete Q32 projection coverage")
    checks = payload.get("exact_value_checks")
    if not isinstance(checks, dict) or not checks or not all(checks.values()):
        raise ValueError("V3 requires exact V2 selected-weight checks")
    return payload, sha256_file(path)


def validate_layer_bank_catalog(
    catalog_path: Path,
    bank_path: Path,
    *,
    layer_id: int,
) -> dict[str, Any]:
    """Validate the trusted full-SHA bank receipt plus current file identity.

    The 1.3-GiB bank is deliberately not reread here.  The conversion catalog
    already contains its complete SHA-256; V3 rechecks its path, size, layer,
    geometry, logical coverage, storage type, and trusted source identity before
    the native loader mmaps the file.
    """

    catalog = json.loads(catalog_path.read_text())
    if catalog.get("storage") != "MXFP4":
        raise ValueError("V3 catalog is not MXFP4")
    banks = catalog.get("banks")
    if not isinstance(banks, list):
        raise ValueError("V3 catalog has no bank list")
    matches = [item for item in banks if item.get("layer") == layer_id]
    if len(matches) != 1:
        raise ValueError("V3 catalog must contain exactly one selected layer")
    item = matches[0]
    if item.get("file") != bank_path.name:
        raise ValueError("V3 bank filename differs from the catalog")
    if item.get("storage") != "MXFP4" or item.get("complete_cpu_backing") is not True:
        raise ValueError("V3 bank is not complete MXFP4 CPU backing")
    if item.get("logical_ids") != list(range(LOGICAL_EXPERTS)):
        raise ValueError("V3 CPU bank does not cover all logical experts")
    if item.get("geometry") != {
        "experts": LOGICAL_EXPERTS,
        "hidden": HIDDEN_SIZE,
        "intermediate": INTERMEDIATE_SIZE,
    }:
        raise ValueError("V3 CPU bank geometry differs from frozen Qwen")
    stat = bank_path.stat()
    if item.get("bytes") != stat.st_size:
        raise ValueError("V3 CPU bank live size differs from the catalog")
    bank_sha = item.get("sha256")
    if not isinstance(bank_sha, str) or len(bank_sha) != 64:
        raise ValueError("V3 CPU bank catalog SHA-256 is invalid")
    source = item.get("source_identity")
    if not isinstance(source, Mapping) or source.get("revision") != (
        "de4b8e4d43b917e7706784d8bb445c9af86a3540"
    ):
        raise ValueError("V3 CPU bank source revision differs from S0")
    return {
        "catalog_path": str(catalog_path),
        "catalog_sha256": sha256_file(catalog_path),
        "bank_path": str(bank_path),
        "bank_size_bytes": stat.st_size,
        "bank_sha256_from_trusted_catalog": bank_sha,
        "bank_identity_method": "TRUSTED_FULL_SHA_RECEIPT_PLUS_LIVE_SIZE",
        "layer_id": layer_id,
    }


__all__ = (
    "HIDDEN_SIZE",
    "HybridRouteCase",
    "INTERMEDIATE_SIZE",
    "LOGICAL_EXPERTS",
    "PHYSICAL_NPU_EXPERTS",
    "Q32_V3_CONTRACT_API",
    "TOP_K",
    "frozen_route_cases",
    "sha256_file",
    "validate_layer_bank_catalog",
    "validate_v2_receipt",
)
