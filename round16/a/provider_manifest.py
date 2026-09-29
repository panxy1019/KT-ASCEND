"""Runtime-neutral manifest for the Round16 SGLang/KT provider boundary."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from round16.common.contracts import CONTRACT_VERSION, Phase, ProviderSpec, SupportDecision


PROVIDER_NAME = "kt-ascend-round16"
OPERATION_QSA = "qsa.lightning_indexer"
OPERATION_GDN = "gdn.rmsnorm_sigmoid_gate"
OPERATION_MOE = "moe.resident_grouped"


_DEPENDENCY_TIERS: dict[str, dict[str, Any]] = {
    "P0": {
        "required": True,
        "imports": ["round16.common", "Python standard library"],
        "purpose": "provider contract, manifest, metadata adaptation, and tests",
    },
    "P1": {
        "required": False,
        "imports": ["sglang", "ktransformers"],
        "purpose": "explicit host-framework registration and fallback integration",
    },
    "P2": {
        "required": False,
        "imports": ["torch", "torch_npu", "CANN", "Ascend device"],
        "purpose": "device execution; excluded from import and offline validation",
        "status": "DEVICE_REQUIRED_NOT_RUN",
    },
}


_OPERATIONS: dict[str, dict[str, Any]] = {
    OPERATION_QSA: {
        "phases": (Phase.DECODE,),
        "state_format": "qsa-index-v1",
        "fallback": "Round14a eager round9.qsa.select_tokens plus promoted sparse GQA",
        "graph_capability": "NOT_ADVERTISED_DEVICE_REQUIRED_NOT_RUN",
        "round15_status": "ROUND15_COMPLETE_SIMPLIFIED_DIRECTIONAL_NO_PROMOTION",
    },
    OPERATION_GDN: {
        "phases": (Phase.PREFILL, Phase.DECODE),
        "state_format": "gdn-state-v1",
        "fallback": "Round14a GDN core plus eager sigmoid_gated_rms_norm",
        "graph_capability": "NOT_ADVERTISED_DEVICE_REQUIRED_NOT_RUN",
        "round15_status": "ROUND15_COMPLETE_SIMPLIFIED_DIRECTIONAL_NO_PROMOTION",
    },
    OPERATION_MOE: {
        "phases": (Phase.PREFILL, Phase.DECODE),
        "state_format": "resident-moe-v1",
        "fallback": "Existing eager Ascend grouped-MoE provider",
        "graph_capability": "IDENTITY_STABLE_CAPTURE_QUALIFICATION_NOT_RUN",
        "round15_status": "NO_NEW_A_CANDIDATE_ALREADY_CURRENT_PROVIDER",
    },
}

_RESOLVED_INPUT = {
    "name": "hidden_states",
    "rank": 2,
    "token_rows": {"minimum": 1, "maximum": 128},
    "hidden_width": 2048,
    "dtype": "bfloat16",
    "layout": "ND",
    "stride": [2048, 1],
    "stride_may_be_omitted": True,
}


def dependency_tiers() -> dict[str, dict[str, Any]]:
    """Return an isolated dependency classification for packaging tools."""
    return deepcopy(_DEPENDENCY_TIERS)


def manifest_payload() -> dict[str, Any]:
    """Return a JSON-serializable provider manifest without probing a runtime."""
    operations = []
    for operation, capability in _OPERATIONS.items():
        item = deepcopy(capability)
        item["operation"] = operation
        item["phases"] = [phase.value for phase in capability["phases"]]
        item["input_spec"] = deepcopy(_RESOLVED_INPUT)
        operations.append(item)
    return {
        "schema": 1,
        "provider_name": PROVIDER_NAME,
        "contract_version": CONTRACT_VERSION,
        "operations": operations,
        "dependency_tiers": dependency_tiers(),
        "registration": "EXPLICIT_IDEMPOTENT",
        "import_side_effects": "NONE",
        "device_validation": "DEVICE_REQUIRED_NOT_RUN",
        "performance_validation": "NOT_RUN",
    }


def decide_support(
    spec: ProviderSpec,
    *,
    backend_available: bool,
) -> SupportDecision:
    """Apply the static provider contract without importing device packages."""
    capability = _OPERATIONS.get(spec.operation)
    if capability is None:
        return SupportDecision.reject(
            "UNSUPPORTED_OPERATION", f"operation {spec.operation!r} is not declared"
        )
    if spec.phase not in capability["phases"]:
        return SupportDecision.reject(
            "UNSUPPORTED_PHASE",
            f"{spec.operation} does not support {spec.phase.value}",
        )
    if spec.metadata_version != CONTRACT_VERSION:
        return SupportDecision.reject(
            "UNSUPPORTED_METADATA_VERSION",
            f"expected {CONTRACT_VERSION}, got {spec.metadata_version}",
        )
    if spec.state_format != capability["state_format"]:
        return SupportDecision.reject(
            "UNSUPPORTED_STATE_FORMAT",
            f"expected {capability['state_format']}, got {spec.state_format}",
        )
    if len(spec.inputs) != 1:
        return SupportDecision.reject(
            "UNSUPPORTED_INPUT_COUNT", "exactly one resolved hidden_states tensor is required"
        )
    tensor = spec.inputs[0]
    if tensor.name != "hidden_states":
        return SupportDecision.reject(
            "UNSUPPORTED_INPUT_NAME", "only the resolved hidden_states input is qualified"
        )
    if tensor.rank != 2:
        return SupportDecision.reject(
            "UNSUPPORTED_RANK", "hidden_states must have rank 2"
        )
    token_rows, hidden_width = tensor.shape
    if not 1 <= token_rows <= 128 or hidden_width != 2048:
        return SupportDecision.reject(
            "UNSUPPORTED_SHAPE",
            "hidden_states must have shape [1..128, 2048]",
        )
    if tensor.dtype != "bfloat16":
        return SupportDecision.reject(
            "UNSUPPORTED_DTYPE", "hidden_states must use bfloat16"
        )
    if tensor.layout != "ND":
        return SupportDecision.reject(
            "UNSUPPORTED_LAYOUT", "only the resolved rank-2 ND layout is qualified"
        )
    if tensor.stride is not None and tensor.stride != (2048, 1):
        return SupportDecision.reject(
            "UNSUPPORTED_STRIDE",
            "hidden_states stride must be omitted or equal to (2048, 1)",
        )
    device_types = {tensor.device_type for tensor in spec.inputs}
    if not device_types.issubset({"npu", "mock"}):
        return SupportDecision.device_required(
            "CPU execution is not an Ascend qualification path; inject a mock backend for offline tests"
        )
    expected_stream = "mock-current-stream" if device_types == {"mock"} else "ascend-current-stream"
    if spec.stream_kind != expected_stream:
        return SupportDecision.reject(
            "UNSUPPORTED_STREAM", f"expected {expected_stream}, got {spec.stream_kind}"
        )
    if not backend_available:
        return SupportDecision.device_required(
            "no explicit P2 backend was injected; device execution remains NOT_RUN"
        )
    return SupportDecision.accept(
        "static contract accepted; numerical and performance qualification are separate"
    )
