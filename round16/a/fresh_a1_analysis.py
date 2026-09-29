"""Strict offline comparator and dependency DAG for fresh Round16 A1 blocks.

The analyzer consumes only two new block bundles.  It never imports a runtime,
probes a device, or consults historical traces.  Candidate gates use direct
Host critical intervals as fractions of a direct Decode phase wall in the same
clock domain.  Parent and child spans are retained as separate observations
and are never added or subtracted.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

from .fresh_a1_profile import BLOCK_IDS, STAGE, block_contract, manifest_payload
from .fresh_a1_snapshot import canonical_json_sha256


STATUS_PENDING = "DEVICE_EVIDENCE_PENDING"
STATUS_PASS = "PASS_TWO_FRESH_BLOCKS"
STATUS_INSUFFICIENT = "INSUFFICIENT_ATTRIBUTION"
STATUS_A1_INSUFFICIENT = "A1_INSUFFICIENT_ATTRIBUTION"

_QSA_LAYERS = (3, 23, 47)
_GDN_LAYERS = (0, 24, 46)
_EXPERT_LAYERS = (0, 24, 47)
_MEASUREMENT_ROLES = ("W1_MEASUREMENT", "W2_MEASUREMENT")
_MINIMUM_DECODE_WALL_FRACTION = 0.10
_CANDIDATE_EVIDENCE_NAMES = (
    "A2_HOST_SYNC_EXPOSURE",
    "A3_HOST_LAUNCH_METADATA",
)

_HOST_COMPONENTS = frozenset(
    {
        "PARENT_ROUTED_EXPERT",
        "HOST_DISPATCH",
        "HOST_SYNC",
        "HOST_MERGE",
        "DECODE_PHASE_WALL",
    }
)
_NPU_COMPONENTS = frozenset(
    {"QSA_INDEXER", "GDN", "NPU_RESIDENT", "NPU_SHARED"}
)
_EXPERT_CHILD_COMPONENTS = frozenset(
    {"HOST_DISPATCH", "HOST_SYNC", "HOST_MERGE", "NPU_RESIDENT", "NPU_SHARED", "CPU_EXPERT"}
)


class EvidenceContractError(RuntimeError):
    """Fresh evidence is incomplete, drifted, or arithmetically unsafe."""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvidenceContractError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvidenceContractError(f"{label} must be a JSON object")
    return value


def load_evidence_bundle(
    receipt_path: str | Path,
    observer_metadata_path: str | Path,
    *,
    max_observer_rows: int = 100_000,
) -> dict[str, Any]:
    """Load a SHA-bound raw receipt and observer JSONL without device access."""
    if type(max_observer_rows) is not int or max_observer_rows <= 0:
        raise ValueError("max_observer_rows must be a positive exact integer")
    raw_path = Path(receipt_path).resolve()
    metadata_path = Path(observer_metadata_path).resolve()
    receipt = _json_object(raw_path, "raw receipt")
    metadata = _json_object(metadata_path, "observer metadata")
    if metadata.get("receipt_sha256") != _file_sha256(raw_path):
        raise EvidenceContractError("raw receipt SHA does not match observer metadata")
    rows_value = metadata.get("rows_path")
    if not isinstance(rows_value, str) or not rows_value:
        raise EvidenceContractError("observer metadata rows path is missing")
    rows_path = Path(rows_value).resolve()
    receipt_rows_value = receipt.get("observer_path")
    if (
        not isinstance(receipt_rows_value, str)
        or Path(receipt_rows_value).resolve() != rows_path
    ):
        raise EvidenceContractError("raw receipt observer path identity drift")
    if metadata.get("rows_sha256") != _file_sha256(rows_path):
        raise EvidenceContractError("observer rows SHA does not match metadata")
    rows: list[dict[str, Any]] = []
    try:
        with rows_path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                if len(rows) >= max_observer_rows:
                    raise EvidenceContractError("observer row limit exceeded")
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise EvidenceContractError(
                        f"invalid observer JSONL row {line_number}: {exc}"
                    ) from exc
                if not isinstance(row, dict):
                    raise EvidenceContractError(
                        f"observer JSONL row {line_number} must be an object"
                    )
                rows.append(row)
    except OSError as exc:
        raise EvidenceContractError(f"cannot read observer rows: {exc}") from exc
    observer = {
        key: metadata[key]
        for key in (
            "schema",
            "synchronize_inserted",
            "query_policy",
            "source_identity",
            "aggregations",
            "candidate_regions",
            "measurement_event_flush",
            "attribution_status",
            "attribution_reason",
        )
        if key in metadata
    }
    observer["rows"] = rows
    return {"receipt": receipt, "observer": observer}


def dependency_dag_payload() -> dict[str, Any]:
    """Return the frozen region DAG and its non-additive arithmetic rules."""
    return {
        "schema": 1,
        "stage": "ROUND16_A1_CRITICAL_PATH",
        "status": "FROZEN_OFFLINE",
        "evidence_input_contract": {
            "loader": "load_evidence_bundle",
            "identity_binding": [
                "receipt_sha256",
                "rows_sha256",
                "observer_path",
            ],
            "receipt": "round16-a1-raw-receipt.json",
            "observer_rows": "SHA-bound JSONL",
            "observer_metadata_required": [
                "receipt_sha256",
                "rows_path",
                "rows_sha256",
                "synchronize_inserted",
                "query_policy",
                "source_identity",
                "aggregations",
            ],
            "candidate_terminal_decision_metadata_required": ["candidate_regions"],
            "max_observer_rows_default": 100_000,
        },
        "nodes": [
            {
                "id": "QSA_INDEXER",
                "clock_domain": "NPU_EVENT_ELAPSED_MS",
                "metric": "direct_event_elapsed_ms",
                "candidate": None,
            },
            {
                "id": "GDN",
                "clock_domain": "NPU_EVENT_ELAPSED_MS",
                "metric": "direct_event_elapsed_ms",
                "candidate": None,
                "candidate_exclusion": "GDN duration is not CPU-result exposed wait",
            },
            {
                "id": "PARENT_ROUTED_EXPERT",
                "clock_domain": "HOST_MONOTONIC_NS",
                "metric": "direct_parent_duration_ns_non_additive",
                "candidate": None,
            },
            {
                "id": "HOST_DISPATCH",
                "clock_domain": "HOST_MONOTONIC_NS",
                "metric": "direct_duration_ns",
                "candidate": None,
            },
            {
                "id": "NPU_EXPERT",
                "components": ["NPU_RESIDENT", "NPU_SHARED"],
                "clock_domain": "NPU_EVENT_ELAPSED_MS",
                "metric": "direct_event_elapsed_ms",
                "candidate": None,
                "candidate_exclusion": "kernel duration is not Host launch or metadata cost",
            },
            {
                "id": "CPU_EXPERT",
                "clock_domain": "NATIVE_WORKER_CLOCK_UNAVAILABLE",
                "metric": None,
                "candidate": None,
                "status": "UNKNOWN_NATIVE_ABI_NO_CALLBACK",
            },
            {
                "id": "HOST_SYNC",
                "clock_domain": "HOST_MONOTONIC_NS",
                "metric": "direct_duration_ns",
                "candidate": None,
            },
            {
                "id": "HOST_MERGE",
                "clock_domain": "HOST_MONOTONIC_NS",
                "metric": "direct_duration_ns",
                "candidate": None,
            },
            {
                "id": "DECODE_PHASE_WALL",
                "clock_domain": "HOST_MONOTONIC_NS",
                "metric": "direct_request_phase_duration_ns",
                "candidate": None,
                "status": "EVIDENCE_REQUIRED",
            },
            {
                "id": "HOST_SYNC_EXPOSED_WAIT",
                "clock_domain": "HOST_MONOTONIC_NS",
                "metric": "direct_critical_path_interval_ns",
                "candidate": "A2",
                "status": "EVIDENCE_REQUIRED",
            },
            {
                "id": "A2_INDEPENDENT_NPU_WINDOW_PROOF",
                "clock_domain": "DEPENDENCY_ONLY_NO_TIME_ARITHMETIC",
                "metric": None,
                "candidate": "A2",
                "status": "EVIDENCE_REQUIRED",
            },
            {
                "id": "HOST_LAUNCH_METADATA_CRITICAL",
                "clock_domain": "HOST_MONOTONIC_NS",
                "metric": "direct_critical_path_interval_ns",
                "candidate": "A3",
                "status": "EVIDENCE_REQUIRED",
            },
        ],
        "edges": [
            {
                "from": "PARENT_ROUTED_EXPERT",
                "to": "HOST_DISPATCH",
                "relation": "CONTAINS_NON_ADDITIVE",
            },
            {
                "from": "PARENT_ROUTED_EXPERT",
                "to": "NPU_EXPERT",
                "relation": "CONTAINS_NON_ADDITIVE",
            },
            {
                "from": "PARENT_ROUTED_EXPERT",
                "to": "CPU_EXPERT",
                "relation": "CONTAINS_NON_ADDITIVE",
            },
            {"from": "HOST_DISPATCH", "to": "CPU_EXPERT", "relation": "HAPPENS_BEFORE"},
            {"from": "CPU_EXPERT", "to": "HOST_SYNC", "relation": "COMPLETES_BEFORE_RETURN"},
            {"from": "NPU_EXPERT", "to": "HOST_MERGE", "relation": "READY_BEFORE_MERGE"},
            {"from": "HOST_SYNC", "to": "HOST_MERGE", "relation": "WAIT_EVENT_BEFORE_MERGE"},
            {
                "from": "DECODE_PHASE_WALL",
                "to": "HOST_SYNC_EXPOSED_WAIT",
                "relation": "CONTAINS_NON_ADDITIVE",
            },
            {
                "from": "HOST_SYNC_EXPOSED_WAIT",
                "to": "A2_INDEPENDENT_NPU_WINDOW_PROOF",
                "relation": "REQUIRES_UNUSED_SAME_LAYER_WINDOW",
            },
            {
                "from": "DECODE_PHASE_WALL",
                "to": "HOST_LAUNCH_METADATA_CRITICAL",
                "relation": "CONTAINS_NON_ADDITIVE",
            },
        ],
        "sample_layers": {
            "QSA_INDEXER": list(_QSA_LAYERS),
            "GDN": list(_GDN_LAYERS),
            "NPU_EXPERT": list(_EXPERT_LAYERS),
        },
        "arithmetic_policy": {
            "operation": "DIRECT_REGION_FRACTION_OF_DECODE_WALL",
            "score_clock_domain": "HOST_MONOTONIC_NS",
            "minimum_decode_wall_fraction": _MINIMUM_DECODE_WALL_FRACTION,
            "parent_child_sum_allowed": False,
            "parent_minus_children_allowed": False,
            "cross_clock_sum_allowed": False,
            "preaggregated_input_allowed": False,
            "cpu_unknown_coercion": "FORBIDDEN",
            "gdn_as_cpu_wait_proxy": "FORBIDDEN",
            "npu_kernel_as_host_launch_proxy": "FORBIDDEN",
        },
        "candidate_gates": {
            "A2": {
                "numerator": "HOST_SYNC_EXPOSED_WAIT",
                "denominator": "DECODE_PHASE_WALL",
                "clock_domain": "HOST_MONOTONIC_NS",
                "minimum_decode_wall_fraction": _MINIMUM_DECODE_WALL_FRACTION,
                "requires_both_fresh_blocks": True,
                "requires_same_layer_independent_npu_window": True,
                "requires_primary_actionable_region_in_each_block": True,
                "cpu_native_compute_may_be_inferred": False,
            },
            "A3": {
                "numerator": "HOST_LAUNCH_METADATA_CRITICAL",
                "denominator": "DECODE_PHASE_WALL",
                "clock_domain": "HOST_MONOTONIC_NS",
                "minimum_decode_wall_fraction": _MINIMUM_DECODE_WALL_FRACTION,
                "requires_both_fresh_blocks": True,
                "requires_outside_resident_only_graph": True,
                "requires_primary_actionable_region_in_each_block": True,
                "npu_kernel_duration_may_be_used": False,
            },
        },
        "candidate_mapping": {
            "A2": "Host sync exposed wait is at least 10% of Decode wall in both fresh blocks and an unused same-layer independent NPU window is proven",
            "A3": "Host launch or metadata critical interval is at least 10% of Decode wall in both fresh blocks and lies outside the resident-only graph",
            "NO_CANDIDATE": "no single gate clears the two-block threshold",
        },
    }


def _pending_payload(
    present: Sequence[str],
    *,
    blocks: Sequence[Mapping[str, Any]] = (),
    missing_candidate_evidence: Sequence[str] = _CANDIDATE_EVIDENCE_NAMES,
) -> dict[str, Any]:
    missing = [item for item in BLOCK_IDS.values() if item not in present]
    return {
        "critical_path": {
            "schema": 1,
            "stage": "ROUND16_A1_CRITICAL_PATH",
            "status": STATUS_PENDING,
            "fresh_blocks_present": list(present),
            "fresh_blocks_required": list(BLOCK_IDS.values()),
            "fresh_blocks_missing": missing,
            "historical_trace_eligible": False,
            "blocks": [dict(item) for item in blocks],
            "missing_candidate_evidence": list(missing_candidate_evidence),
            "arithmetic": {
                "operation": "DIRECT_REGION_FRACTION_OF_DECODE_WALL",
                "clock_domain": "HOST_MONOTONIC_NS",
                "added_terms": [],
                "cross_clock_additions": 0,
                "nested_span_additions": 0,
            },
            "candidate_decision_status": STATUS_PENDING,
            "execution": "DEVICE_REQUIRED_NOT_RUN",
        },
        "dag": dependency_dag_payload(),
        "decision": {
            "schema": 1,
            "stage": "ROUND16_A_CANDIDATE_DECISION",
            "status": STATUS_PENDING,
            "selected_candidate": None,
            "fresh_blocks_present": list(present),
            "fresh_blocks_required": list(BLOCK_IDS.values()),
            "historical_evidence_used": False,
            "missing_candidate_evidence": list(missing_candidate_evidence),
            "allowed_terminal_candidates": ["A2", "A3", "NO_CANDIDATE"],
            "candidate_enabled": False,
            "execution": "DEVICE_REQUIRED_NOT_RUN",
        },
    }


def _exact_request_identity(
    block_id: str,
    item: Mapping[str, Any],
    request_id: str,
) -> dict[str, Any]:
    return {
        "block_id": block_id,
        "request_id": request_id,
        "role": item["role"],
        "input_tokens": item["input_tokens"],
        "max_new_tokens": (
            item["max_new_tokens"] if "max_new_tokens" in item else item["output_tokens"]
        ),
        "ignore_eos": item.get("ignore_eos", False),
        "fixture_sha256": item.get("fixture_sha256"),
        "input_ids_sha256": item.get("input_ids_sha256"),
        "messages_sha256": item.get("messages_sha256"),
    }


def _nonempty_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise EvidenceContractError(f"{label} must be a non-empty mapping")
    result = dict(value)
    if any(not isinstance(key, str) or not key for key in result):
        raise EvidenceContractError(f"{label} keys must be non-empty strings")
    return result


def _validate_receipt(
    bundle: Mapping[str, Any],
    *,
    block: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    block_id = BLOCK_IDS[block]
    receipt = _nonempty_mapping(bundle.get("receipt"), f"block {block} receipt")
    observer = _nonempty_mapping(bundle.get("observer"), f"block {block} observer")
    contract = block_contract(block)
    if receipt.get("stage") != STAGE:
        raise EvidenceContractError(f"block {block} stage identity drift")
    if receipt.get("status") != "RAW_DEVICE_RESULT_UNREVIEWED":
        raise EvidenceContractError(f"block {block} raw receipt status is not reviewable")
    if receipt.get("block_id") != block_id or receipt.get("snapshot_id") != contract.snapshot_id:
        raise EvidenceContractError(f"block {block} immutable identity drift")
    if receipt.get("candidate_selected") is not None:
        raise EvidenceContractError("raw receipt cannot contain a preselected candidate")
    source_manifest = receipt.get("source_manifest_sha256")
    if not isinstance(source_manifest, str) or len(source_manifest) != 64:
        raise EvidenceContractError(f"block {block} source manifest identity missing")

    schedule = manifest_payload()["schedule"]
    schedule_sha = canonical_json_sha256(schedule)
    if receipt.get("schedule_sha256") != schedule_sha:
        raise EvidenceContractError(f"block {block} schedule identity drift")
    generations = receipt.get("generations")
    if (
        receipt.get("generation_attempts") != 6
        or not isinstance(generations, list)
        or len(generations) != 6
    ):
        raise EvidenceContractError(f"block {block} must contain exactly six requests")

    request_roles: dict[str, str] = {}
    source_identity: dict[str, Any] | None = None
    for item, generation in zip(schedule, generations, strict=True):
        if not isinstance(generation, Mapping):
            raise EvidenceContractError(f"block {block} generation row must be a mapping")
        request_id = f"{block_id}:{item['ordinal']}:{item['role']}"
        if (
            generation.get("ordinal") != item["ordinal"]
            or generation.get("role") != item["role"]
            or generation.get("request_id") != request_id
            or generation.get("status") != "COMPLETE"
        ):
            raise EvidenceContractError(f"block {block} request identity/order drift")
        if generation.get("schedule") != item:
            raise EvidenceContractError(f"block {block} token or schedule identity drift")
        driver = _nonempty_mapping(
            generation.get("driver"), f"block {block} driver receipt"
        )
        if driver.get("status") != "COMPLETE":
            raise EvidenceContractError(f"block {block} driver request is incomplete")
        expected_request = _exact_request_identity(block_id, item, request_id)
        if driver.get("observed_request") != expected_request:
            raise EvidenceContractError(f"block {block} token or request identity drift")
        observed_source = _nonempty_mapping(
            driver.get("source_identity"), f"block {block} source identity"
        )
        if source_identity is None:
            source_identity = observed_source
        elif observed_source != source_identity:
            raise EvidenceContractError(f"block {block} source identity drift between requests")
        request_roles[request_id] = item["role"]
    assert source_identity is not None
    if observer.get("source_identity") != source_identity:
        raise EvidenceContractError(f"block {block} observer/source identity drift")
    return receipt, observer, request_roles


def _exact_int(value: Any, label: str) -> int:
    if type(value) is not int:
        raise EvidenceContractError(f"{label} must be an exact integer")
    return value


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceContractError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise EvidenceContractError(f"{label} must be finite and non-negative")
    return result


def _validate_aggregations(observer: Mapping[str, Any], rows_by_span: Mapping[str, Mapping[str, Any]]) -> None:
    groups = observer.get("aggregations", [])
    if not isinstance(groups, list):
        raise EvidenceContractError("observer aggregations must be a list")
    for group in groups:
        if not isinstance(group, Mapping) or group.get("operation") != "SUM":
            raise EvidenceContractError("only explicit SUM aggregation declarations are recognized")
        terms = group.get("terms")
        if not isinstance(terms, list) or len(terms) < 2:
            raise EvidenceContractError("aggregation requires at least two terms")
        if len(terms) != len(set(terms)):
            raise EvidenceContractError("duplicate aggregation terms are forbidden")
        if any(term not in rows_by_span for term in terms):
            raise EvidenceContractError("aggregation references an unknown event")
        for left in terms:
            for right in terms:
                if rows_by_span[right].get("parent_span_id") == left:
                    raise EvidenceContractError("nested parent/child aggregation is forbidden")
        clocks = {rows_by_span[term]["clock_domain"] for term in terms}
        if len(clocks) != 1:
            raise EvidenceContractError("cross-clock aggregation is forbidden")
        raise EvidenceContractError("pre-aggregated evidence is forbidden")


def _candidate_source_spans(
    value: Any,
    *,
    label: str,
    request_id: str,
    rows_by_span: Mapping[str, Mapping[str, Any]],
    components: frozenset[str],
) -> list[Mapping[str, Any]]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise EvidenceContractError(f"{label} source spans must be non-empty identities")
    if len(value) != len(set(value)):
        raise EvidenceContractError(f"{label} duplicate source span identity")
    try:
        rows = [rows_by_span[item] for item in value]
    except KeyError as exc:
        raise EvidenceContractError(f"{label} references an unknown source span") from exc
    for row in rows:
        if row["request_id"] != request_id or row["component"] not in components:
            raise EvidenceContractError(f"{label} source span identity drift")
    identities = set(value)
    if any(row["parent_span_id"] in identities for row in rows):
        raise EvidenceContractError(f"{label} nested parent/child source spans are forbidden")
    return rows


def _validate_candidate_regions(
    observer: Mapping[str, Any],
    *,
    rows_by_span: Mapping[str, Mapping[str, Any]],
    measurement_request_ids: frozenset[str],
) -> dict[str, Any]:
    regions = observer.get("candidate_regions", [])
    if not isinstance(regions, list):
        raise EvidenceContractError("candidate regions must be a list")
    attribution_status = observer.get("attribution_status")
    if attribution_status not in (None, "COMPLETE_DIRECT_REGIONS", STATUS_INSUFFICIENT):
        raise EvidenceContractError("candidate attribution status is unsupported")
    if attribution_status == STATUS_INSUFFICIENT:
        if regions:
            raise EvidenceContractError(
                "insufficient attribution cannot contain candidate regions"
            )
        reason = observer.get("attribution_reason")
        if not isinstance(reason, str) or not reason:
            raise EvidenceContractError("insufficient attribution reason is missing")
        walls = [
            row
            for row in rows_by_span.values()
            if row["request_id"] in measurement_request_ids
            and row["component"] == "DECODE_PHASE_WALL"
        ]
        if {row["request_id"] for row in walls} != set(measurement_request_ids):
            raise EvidenceContractError(
                "insufficient attribution still requires each direct Decode wall"
            )
        if any(
            row.get("derivation")
            != "DIRECT_SERIAL_SCHEDULER_BOUNDING_INTERVAL"
            for row in walls
        ):
            raise EvidenceContractError(
                "insufficient attribution Decode wall derivation drift"
            )
        return {
            "status": STATUS_INSUFFICIENT,
            "reason": reason,
            "missing": [],
            "minimum_decode_wall_fraction": _MINIMUM_DECODE_WALL_FRACTION,
            "median_fraction_of_decode_wall": {"A2": None, "A3": None},
            "per_request": {"A2": {}, "A3": {}},
            "gate_met": {"A2": False, "A3": False},
        }
    seen: set[tuple[str, str]] = set()
    region_ids: set[str] = set()
    ratios: dict[str, list[float]] = {"A2": [], "A3": []}
    per_request: dict[str, dict[str, Any]] = {"A2": {}, "A3": {}}
    decode_wall_by_request: dict[str, str] = {}
    for raw in regions:
        if not isinstance(raw, Mapping):
            raise EvidenceContractError("candidate region must be a mapping")
        region = dict(raw)
        candidate = region.get("candidate")
        expected = {
            "A2": ("HOST_SYNC_EXPOSED_WAIT", frozenset({"HOST_SYNC"})),
            "A3": (
                "HOST_LAUNCH_METADATA_CRITICAL",
                frozenset({"HOST_DISPATCH"}),
            ),
        }
        if candidate not in expected:
            raise EvidenceContractError("candidate region has an unsupported candidate")
        component, source_components = expected[candidate]
        request_id = region.get("request_id")
        if request_id not in measurement_request_ids:
            raise EvidenceContractError("candidate region must identify a measurement request")
        identity = (candidate, request_id)
        if identity in seen:
            raise EvidenceContractError("duplicate candidate region identity")
        seen.add(identity)
        region_id = region.get("region_id")
        if not isinstance(region_id, str) or not region_id or region_id in region_ids:
            raise EvidenceContractError("candidate region identity is missing or duplicate")
        region_ids.add(region_id)
        if (
            region.get("component") != component
            or region.get("clock_domain") != "HOST_MONOTONIC_NS"
            or region.get("phase") != "DECODE"
            or region.get("derivation") != "DIRECT_CRITICAL_PATH_INTERVAL_NO_SUM"
        ):
            raise EvidenceContractError(f"{candidate} candidate region contract drift")
        if region.get("aggregation") is not None or region.get("terms") is not None:
            raise EvidenceContractError("candidate region cannot contain an aggregation")
        duration = _exact_int(region.get("duration_ns"), "candidate region duration")
        start_ns = _exact_int(region.get("start_ns"), "candidate region start")
        end_ns = _exact_int(region.get("end_ns"), "candidate region end")
        decode_wall = _exact_int(
            region.get("decode_phase_wall_ns"), "Decode phase wall duration"
        )
        if (
            start_ns < 0
            or end_ns < start_ns
            or end_ns - start_ns != duration
            or duration < 0
            or decode_wall <= 0
            or duration > decode_wall
        ):
            raise EvidenceContractError("candidate region/Decode wall duration is invalid")
        decode_wall_span_id = region.get("decode_phase_wall_span_id")
        decode_wall_row = rows_by_span.get(decode_wall_span_id)
        if (
            decode_wall_row is None
            or decode_wall_row.get("component") != "DECODE_PHASE_WALL"
            or decode_wall_row.get("request_id") != request_id
            or decode_wall_row.get("clock_domain") != "HOST_MONOTONIC_NS"
            or decode_wall_row.get("duration_ns") != decode_wall
            or decode_wall_row.get("derivation") != "DIRECT_OBSERVATION"
            or start_ns < decode_wall_row.get("start_ns", -1)
            or end_ns > decode_wall_row.get("end_ns", -1)
        ):
            raise EvidenceContractError(
                "candidate region lacks its direct same-clock Decode phase wall"
            )
        previous_wall = decode_wall_by_request.setdefault(
            str(request_id), str(decode_wall_span_id)
        )
        if previous_wall != decode_wall_span_id:
            raise EvidenceContractError(
                "candidate regions disagree on the Decode phase wall identity"
            )
        source_rows = _candidate_source_spans(
            region.get("source_span_ids"),
            label=str(candidate),
            request_id=str(request_id),
            rows_by_span=rows_by_span,
            components=source_components,
        )
        if any(row["clock_domain"] != "HOST_MONOTONIC_NS" for row in source_rows):
            raise EvidenceContractError("candidate region source crosses clock domains")
        ratio = duration / decode_wall
        detail: dict[str, Any] = {
            "request_id": request_id,
            "fraction_of_decode_wall": ratio,
            "clock_domain": "HOST_MONOTONIC_NS",
            "arithmetic": "DIRECT_INTERVAL_DIVIDED_BY_DIRECT_WALL_NO_SUM",
        }
        if candidate == "A2":
            proof = region.get("dependency_proof")
            if not isinstance(proof, Mapping):
                raise EvidenceContractError("A2 independent NPU window proof is missing")
            if (
                proof.get("type") != "SAME_LAYER_INDEPENDENT_NPU_WINDOW"
                or proof.get("status") not in ("AVAILABLE_UNUSED", "NOT_PROVEN")
                or proof.get("layers") != list(_EXPERT_LAYERS)
                or proof.get("target") != "CPU_EXPERT"
                or proof.get("clock_relation")
                != "DEPENDENCY_ONLY_NO_TIME_ARITHMETIC"
            ):
                raise EvidenceContractError("A2 independent NPU window proof drift")
            npu_rows = _candidate_source_spans(
                proof.get("source_span_ids"),
                label="A2 dependency proof",
                request_id=str(request_id),
                rows_by_span=rows_by_span,
                components=frozenset({"NPU_RESIDENT", "NPU_SHARED"}),
            )
            if {row["layer"] for row in npu_rows} != set(_EXPERT_LAYERS):
                raise EvidenceContractError("A2 proof lacks each required same-layer NPU window")
            detail["independent_npu_window"] = proof["status"]
            detail["gate_met"] = (
                ratio >= _MINIMUM_DECODE_WALL_FRACTION
                and proof["status"] == "AVAILABLE_UNUSED"
            )
        else:
            if type(region.get("outside_resident_only_graph")) is not bool:
                raise EvidenceContractError("A3 resident-only graph boundary is missing")
            detail["outside_resident_only_graph"] = region[
                "outside_resident_only_graph"
            ]
            detail["gate_met"] = (
                ratio >= _MINIMUM_DECODE_WALL_FRACTION
                and region["outside_resident_only_graph"]
            )
        ratios[candidate].append(ratio)
        per_request[candidate][str(request_id)] = detail

    missing = []
    for candidate, evidence_name in zip(
        ("A2", "A3"), _CANDIDATE_EVIDENCE_NAMES, strict=True
    ):
        if set(per_request[candidate]) != set(measurement_request_ids):
            missing.append(evidence_name)
    if regions:
        for request_id in measurement_request_ids:
            walls = [
                row
                for row in rows_by_span.values()
                if row["request_id"] == request_id
                and row["component"] == "DECODE_PHASE_WALL"
            ]
            if len(walls) != 1:
                raise EvidenceContractError(
                    "each candidate measurement needs exactly one Decode phase wall"
                )
    complete = not missing
    median_ratios = {
        candidate: (float(median(items)) if items else None)
        for candidate, items in ratios.items()
    }
    gates = {
        candidate: complete
        and all(item["gate_met"] for item in per_request[candidate].values())
        for candidate in ("A2", "A3")
    }
    if complete:
        gates["A2"] = gates["A2"] and median_ratios["A2"] > median_ratios["A3"]
        gates["A3"] = gates["A3"] and median_ratios["A3"] > median_ratios["A2"]
    return {
        "status": "COMPLETE" if complete else STATUS_PENDING,
        "missing": missing,
        "minimum_decode_wall_fraction": _MINIMUM_DECODE_WALL_FRACTION,
        "median_fraction_of_decode_wall": median_ratios,
        "per_request": per_request,
        "gate_met": gates,
    }


def _validate_rows(
    observer: Mapping[str, Any],
    *,
    block: str,
    request_roles: Mapping[str, str],
) -> dict[str, Any]:
    if observer.get("synchronize_inserted") is not False:
        raise EvidenceContractError("observer synchronization must be exactly false")
    if observer.get("query_policy") != "DEFERRED_QUERY_NO_SYNCHRONIZE":
        raise EvidenceContractError("observer query policy drift or synchronization risk")
    rows = observer.get("rows")
    if not isinstance(rows, list) or not rows:
        raise EvidenceContractError(f"block {block} observer rows are missing")
    block_id = BLOCK_IDS[block]
    spans: dict[str, Mapping[str, Any]] = {}
    event_ids: set[str] = set()
    measurements = frozenset(
        request_id
        for request_id, role in request_roles.items()
        if role in _MEASUREMENT_ROLES
    )
    values: dict[str, list[float]] = {"QSA_INDEXER": [], "GDN": [], "NPU_EXPERT": []}
    measurement_event_flush: dict[str, Any] | None = None
    if observer.get("attribution_status") == STATUS_INSUFFICIENT:
        flush = observer.get("measurement_event_flush")
        if not isinstance(flush, Mapping):
            raise EvidenceContractError("measurement event flush evidence is missing")
        measurement_event_flush = dict(flush)
        if (
            measurement_event_flush.get("status")
            != "COMPLETE_MEASUREMENT_BATCHES_SERIALIZED"
            or measurement_event_flush.get("request_ids") != sorted(measurements)
            or measurement_event_flush.get("required_direct_device_components")
            != ["QSA_INDEXER", "GDN", "NPU_RESIDENT"]
            or measurement_event_flush.get("trailing_schedule_roles")
            != ["SHORT_DRAIN", "FIXED_CANARY"]
            or measurement_event_flush.get("serialized_witness_role") != "SHORT_DRAIN"
            or measurement_event_flush.get("serialized_witness_request_id")
            not in request_roles
            or request_roles.get(
                measurement_event_flush.get("serialized_witness_request_id")
            )
            != "SHORT_DRAIN"
            or measurement_event_flush.get("final_canary_device_completion_required")
            is not False
        ):
            raise EvidenceContractError("measurement event flush evidence drift")

    for raw in rows:
        if not isinstance(raw, Mapping):
            raise EvidenceContractError("observer event must be a mapping")
        row = dict(raw)
        if row.get("block_id") != block_id:
            raise EvidenceContractError(f"block {block} observer block identity drift")
        request_id = row.get("request_id")
        if request_id not in request_roles or "unresolved-scheduler-batch" in str(request_id):
            raise EvidenceContractError(f"block {block} observer request identity drift")
        component = row.get("component")
        if component not in _HOST_COMPONENTS | _NPU_COMPONENTS | {"CPU_EXPERT"}:
            raise EvidenceContractError(f"unsupported observer component {component!r}")
        layer = _exact_int(row.get("layer"), "observer layer")
        for field in (
            "span_id",
            "parent_span_id",
            "stream_id",
            "start_event_id",
            "end_event_id",
            "correlation_id",
            "clock_domain",
        ):
            if not isinstance(row.get(field), str) or not row[field]:
                raise EvidenceContractError(f"observer event identity missing field {field}")
        if row["start_event_id"] == row["end_event_id"]:
            raise EvidenceContractError("observer event identity start/end collision")
        span_id = row["span_id"]
        if span_id in spans:
            raise EvidenceContractError("duplicate observer span identity")
        spans[span_id] = row
        for event_field in ("start_event_id", "end_event_id"):
            event_id = row[event_field]
            if event_id in event_ids:
                raise EvidenceContractError("duplicate observer event identity")
            event_ids.add(event_id)
        if row.get("aggregation") is not None or row.get("derived_from_span_ids"):
            raise EvidenceContractError("derived or nested addition is forbidden")

        clock = row["clock_domain"]
        if component in _HOST_COMPONENTS:
            if clock != "HOST_MONOTONIC_NS":
                raise EvidenceContractError("Host event clock-domain drift")
            start_ns = _exact_int(row.get("start_ns"), "Host start")
            end_ns = _exact_int(row.get("end_ns"), "Host end")
            duration = _exact_int(row.get("duration_ns"), "Host duration")
            if (
                start_ns < 0
                or end_ns < start_ns
                or end_ns - start_ns != duration
                or row.get("elapsed_ms") is not None
            ):
                raise EvidenceContractError("Host event duration fields are invalid")
            allowed_derivation = (
                component == "DECODE_PHASE_WALL"
                and observer.get("attribution_status") == STATUS_INSUFFICIENT
                and row.get("derivation")
                == "DIRECT_SERIAL_SCHEDULER_BOUNDING_INTERVAL"
            )
            if row.get("derivation") != "DIRECT_OBSERVATION" and not allowed_derivation:
                raise EvidenceContractError("Host event must be a direct observation")
        elif component in _NPU_COMPONENTS:
            if clock != "NPU_EVENT_ELAPSED_MS":
                raise EvidenceContractError("NPU event clock-domain drift")
            if any(row.get(field) is not None for field in ("start_ns", "end_ns", "duration_ns")):
                raise EvidenceContractError("NPU event cannot claim Host timestamps")
            elapsed = _finite_number(row.get("elapsed_ms"), "NPU elapsed_ms")
            if (
                row.get("status") != "COMPLETE"
                or row.get("query_policy") != "DEFERRED_QUERY_NO_SYNCHRONIZE"
                or row.get("derivation") != "DIRECT_OBSERVATION"
            ):
                raise EvidenceContractError("NPU event is incomplete or not direct")
            if request_id in measurements:
                key = "NPU_EXPERT" if component in ("NPU_RESIDENT", "NPU_SHARED") else component
                values[key].append(elapsed)
        else:
            if clock != "NATIVE_WORKER_CLOCK_UNAVAILABLE":
                raise EvidenceContractError("CPU Expert clock-domain drift")
            if row.get("status") != "UNKNOWN_NATIVE_ABI_NO_CALLBACK":
                raise EvidenceContractError("CPU Expert must remain UNKNOWN_NATIVE_ABI_NO_CALLBACK")
            if any(
                row.get(field) is not None
                for field in ("start_ns", "end_ns", "duration_ns", "elapsed_ms")
            ):
                raise EvidenceContractError("CPU Expert UNKNOWN cannot be estimated as zero")
            if row.get("derivation") != "NOT_INFERRED_FROM_HOST_BOOKENDS":
                raise EvidenceContractError("CPU Expert UNKNOWN derivation drift")

    _validate_aggregations(observer, spans)

    observed_request_ids = {row["request_id"] for row in spans.values()}
    missing_requests = set(request_roles) - observed_request_ids
    if missing_requests:
        raise EvidenceContractError(
            f"observer events missing for requests {sorted(missing_requests)}"
        )

    for span in spans.values():
        if span["component"] not in _EXPERT_CHILD_COMPONENTS:
            continue
        parent = spans.get(span["parent_span_id"])
        if (
            parent is None
            or parent.get("component") != "PARENT_ROUTED_EXPERT"
            or parent.get("request_id") != span["request_id"]
            or parent.get("layer") != span["layer"]
            or parent.get("correlation_id") != span["correlation_id"]
        ):
            raise EvidenceContractError("expert child event lacks its exact non-additive parent")

    for request_id in request_roles:
        for component, layers in (
            ("PARENT_ROUTED_EXPERT", _EXPERT_LAYERS),
            ("HOST_DISPATCH", _EXPERT_LAYERS),
            ("HOST_SYNC", _EXPERT_LAYERS),
            ("HOST_MERGE", _EXPERT_LAYERS),
            ("CPU_EXPERT", _EXPERT_LAYERS),
        ):
            for layer in layers:
                if not any(
                    row["request_id"] == request_id
                    and row["component"] == component
                    and row["layer"] == layer
                    for row in spans.values()
                ):
                    raise EvidenceContractError(
                        f"{component} missing required layer {layer} for {request_id}"
                    )
    for request_id in measurements:
        for component, layers in (
            ("QSA_INDEXER", _QSA_LAYERS),
            ("GDN", _GDN_LAYERS),
        ):
            for layer in layers:
                if not any(
                    row["request_id"] == request_id
                    and row["component"] == component
                    and row["layer"] == layer
                    for row in spans.values()
                ):
                    raise EvidenceContractError(
                        f"{component} missing required measurement layer {layer} for {request_id}"
                    )
        for layer in _EXPERT_LAYERS:
            if not any(
                row["request_id"] == request_id
                and row["component"] in ("NPU_RESIDENT", "NPU_SHARED")
                and row["layer"] == layer
                for row in spans.values()
            ):
                raise EvidenceContractError(
                    f"NPU expert event missing required measurement layer {layer} for {request_id}"
                )

    if any(not values[key] for key in values):
        raise EvidenceContractError(f"block {block} candidate event values are missing")
    candidate_regions = _validate_candidate_regions(
        observer,
        rows_by_span=spans,
        measurement_request_ids=measurements,
    )
    return {
        "block": block,
        "block_id": block_id,
        "row_count": len(rows),
        "measurement_request_count": len(measurements),
        "measurement_event_flush": measurement_event_flush,
        "diagnostic_direct_npu_event_medians_ms_not_candidate_scores": {
            key: float(median(items)) for key, items in values.items()
        },
        "candidate_regions": candidate_regions,
        "cpu_expert": {
            "status": "UNKNOWN_NATIVE_ABI_NO_CALLBACK",
            "duration": None,
            "included_in_candidate_score": False,
        },
    }


def analysis_payload(
    block_a: Mapping[str, Any] | None = None,
    block_b: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate two fresh blocks and return critical path, DAG, and decision."""
    present: list[str] = []
    validated: list[
        tuple[dict[str, Any], dict[str, Any], dict[str, str], dict[str, Any]]
    ] = []
    for block, bundle in (("A", block_a), ("B", block_b)):
        if bundle is None:
            continue
        if not isinstance(bundle, Mapping):
            raise EvidenceContractError(f"block {block} evidence must be a mapping")
        receipt, observer, request_roles = _validate_receipt(bundle, block=block)
        summary = _validate_rows(
            observer,
            block=block,
            request_roles=request_roles,
        )
        present.append(BLOCK_IDS[block])
        validated.append((receipt, observer, request_roles, summary))
    if len(validated) < 2:
        summaries = [item[3] for item in validated]
        missing = sorted(
            {
                evidence
                for summary in summaries
                for evidence in summary["candidate_regions"]["missing"]
            }
            or set(_CANDIDATE_EVIDENCE_NAMES)
        )
        return _pending_payload(present, blocks=summaries, missing_candidate_evidence=missing)

    source_a = validated[0][1]["source_identity"]
    source_b = validated[1][1]["source_identity"]
    if source_a != source_b:
        raise EvidenceContractError("fresh blocks have source identity drift")
    summaries = [validated[0][3], validated[1][3]]
    missing = sorted(
        {
            evidence
            for summary in summaries
            for evidence in summary["candidate_regions"]["missing"]
        }
    )
    if missing:
        return _pending_payload(
            present,
            blocks=summaries,
            missing_candidate_evidence=missing,
        )

    per_block_gates = {
        summary["block_id"]: dict(summary["candidate_regions"]["gate_met"])
        for summary in summaries
    }
    insufficient = any(
        summary["candidate_regions"]["status"] == STATUS_INSUFFICIENT
        for summary in summaries
    )
    a2_current = not insufficient and all(item["A2"] for item in per_block_gates.values())
    a3_current = not insufficient and all(item["A3"] for item in per_block_gates.values())
    gate_results = {
        "A2": "CURRENT_TWO_BLOCK_A2" if a2_current else "NO_CURRENT_A2",
        "A3": "CURRENT_TWO_BLOCK_A3" if a3_current else "NO_CURRENT_A3",
    }
    if insufficient:
        selected = None
        decision_status = STATUS_A1_INSUFFICIENT
        gate_results = {
            "A2": "NOT_SELECTED_EVIDENCE_GAP",
            "A3": "NOT_SELECTED_EVIDENCE_GAP",
        }
    elif a2_current and not a3_current:
        selected = "A2"
        decision_status = "SELECT_A2"
    elif a3_current and not a2_current:
        selected = "A3"
        decision_status = "SELECT_A3"
    else:
        selected = "NO_CANDIDATE"
        decision_status = "NO_CANDIDATE"

    critical = {
        "schema": 1,
        "stage": "ROUND16_A1_CRITICAL_PATH",
        "status": STATUS_A1_INSUFFICIENT if insufficient else STATUS_PASS,
        "fresh_blocks_present": list(BLOCK_IDS.values()),
        "historical_trace_eligible": False,
        "source_identity": dict(source_a),
        "blocks": summaries,
        "per_block_candidate_gates": per_block_gates,
        "gate_results": gate_results,
        "arithmetic": {
            "operation": "DIRECT_REGION_FRACTION_OF_DECODE_WALL",
            "clock_domain": "HOST_MONOTONIC_NS",
            "minimum_decode_wall_fraction": _MINIMUM_DECODE_WALL_FRACTION,
            "added_terms": [],
            "cross_clock_additions": 0,
            "nested_span_additions": 0,
            "parent_minus_children": False,
        },
        "unknown_regions": [
            {
                "component": "CPU_EXPERT",
                "status": "UNKNOWN_NATIVE_ABI_NO_CALLBACK",
                "duration": None,
                "candidate_score_contribution": None,
            }
        ],
        "candidate_decision_status": decision_status,
    }
    decision = {
        "schema": 1,
        "stage": "ROUND16_A_CANDIDATE_DECISION",
        "status": decision_status,
        "selected_candidate": selected,
        "fresh_blocks_present": list(BLOCK_IDS.values()),
        "fresh_blocks_required": list(BLOCK_IDS.values()),
        "historical_evidence_used": False,
        "decision_clock_domain": "HOST_MONOTONIC_NS",
        "decision_operation": "DIRECT_REGION_FRACTION_OF_DECODE_WALL_NO_SUM",
        "minimum_decode_wall_fraction": _MINIMUM_DECODE_WALL_FRACTION,
        "per_block_candidate_gates": per_block_gates,
        "gate_results": gate_results,
        "candidate_enabled": False,
        "requires_separate_device_review": True,
    }
    return {"critical_path": critical, "dag": dependency_dag_payload(), "decision": decision}


def render_critical_path_markdown(critical_path: Mapping[str, Any]) -> str:
    """Render a concise review companion without changing evidence."""
    status = critical_path.get("status", "INVALID")
    lines = [
        "# Round16-A A1 critical path",
        "",
        f"Status: `{status}`",
        "",
        "Arithmetic: direct critical-region duration divided by direct Decode wall duration in `HOST_MONOTONIC_NS`; parent/child and cross-clock sums are forbidden.",
        "",
    ]
    if status == STATUS_PENDING:
        lines.extend(
            [
                "Two new immutable fresh blocks are required. Historical traces are ineligible.",
                "",
                "Missing candidate evidence: `"
                + "`, `".join(critical_path.get("missing_candidate_evidence", []))
                + "`.",
                "",
                f"Decision: `{critical_path.get('candidate_decision_status')}`",
            ]
        )
        return "\n".join(lines) + "\n"
    if critical_path.get("candidate_decision_status") == STATUS_A1_INSUFFICIENT:
        lines.extend(
            [
                "Both fresh blocks are complete, but the current observer cannot derive a direct A2 or A3 critical interval without adding or subtracting nested spans.",
                "",
                "Decision: `A1_INSUFFICIENT_ATTRIBUTION`; candidate: `null`; A2 and A3 remain `NOT_SELECTED_EVIDENCE_GAP`.",
                "",
                "CPU Expert remains `UNKNOWN_NATIVE_ABI_NO_CALLBACK`.",
            ]
        )
        return "\n".join(lines) + "\n"
    lines.extend(
        [
            "| Block | A2 Host sync / Decode wall | A3 Host launch+metadata / Decode wall |",
            "|---|---:|---:|",
        ]
    )
    for block in critical_path.get("blocks", []):
        scores = block["candidate_regions"]["median_fraction_of_decode_wall"]
        lines.append(
            f"| {block['block_id']} | {scores['A2']:.6f} | {scores['A3']:.6f} |"
        )
    lines.extend(
        [
            "",
            f"Decision: `{critical_path.get('candidate_decision_status')}`",
            "",
            "CPU Expert: `UNKNOWN_NATIVE_ABI_NO_CALLBACK`; duration is null and contributes no numeric score.",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_analysis_outputs(
    output_dir: str | Path,
    block_a: Mapping[str, Any] | None = None,
    block_b: Mapping[str, Any] | None = None,
) -> dict[str, Path]:
    """Write the three JSON review artifacts and the critical-path Markdown."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    payload = analysis_payload(block_a, block_b)
    paths = {
        "critical_path_json": root / "A1_CRITICAL_PATH.json",
        "critical_path_markdown": root / "A1_CRITICAL_PATH.md",
        "dependency_dag_json": root / "A_REGION_DEPENDENCY_DAG.json",
        "candidate_decision_json": root / "A_CANDIDATE_DECISION.json",
    }
    _write_json(paths["critical_path_json"], payload["critical_path"])
    paths["critical_path_markdown"].write_text(
        render_critical_path_markdown(payload["critical_path"]),
        encoding="utf-8",
    )
    _write_json(paths["dependency_dag_json"], payload["dag"])
    _write_json(paths["candidate_decision_json"], payload["decision"])
    return paths


__all__ = (
    "EvidenceContractError",
    "analysis_payload",
    "dependency_dag_payload",
    "load_evidence_bundle",
    "render_critical_path_markdown",
    "write_analysis_outputs",
)
