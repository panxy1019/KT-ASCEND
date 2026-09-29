"""Normalize verified v295/Round14a attribution into strict A1 evidence.

The normalizer copies direct observations and adds stable identities.  It does
not add or subtract nested spans and therefore does not manufacture A2/A3
critical intervals.  When only the frozen Round13 observer and A1 expert
bookends are available, the resulting disposition is explicitly
``INSUFFICIENT_ATTRIBUTION``.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .fresh_a1_analysis import STATUS_INSUFFICIENT


HOST_LAYERS = tuple(range(48))
DEVICE_SAMPLE_LAYERS = (0, 24, 47)
GDN_LAYERS = (0, 24, 46)
QSA_LAYERS = (3, 23, 47)
_HOST_COMPONENTS = (
    "PARENT_ROUTED_EXPERT",
    "HOST_DISPATCH",
    "HOST_SYNC",
    "HOST_MERGE",
    "CPU_EXPERT",
)
_ROUND13_BINDINGS = {
    "GDN": list(GDN_LAYERS),
    "QSA_ATTENTION": list(QSA_LAYERS),
    "QSA_INDEXER": list(QSA_LAYERS),
    "PLE": [1],
    "EXPERT": list(DEVICE_SAMPLE_LAYERS),
}
_EXPECTED_ROLES = (
    "W1_WARMUP",
    "W2_WARMUP",
    "W1_MEASUREMENT",
    "W2_MEASUREMENT",
    "SHORT_DRAIN",
    "FIXED_CANARY",
)


class EvidenceNormalizationError(RuntimeError):
    """Raw observers cannot form a complete fail-closed A1 analysis input."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvidenceNormalizationError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvidenceNormalizationError(f"{label} must be a JSON object")
    return value


def _jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise EvidenceNormalizationError(f"cannot read {label}: {exc}") from exc
    if raw and not raw.endswith(b"\n"):
        raise EvidenceNormalizationError(f"unterminated {label}")
    rows = []
    for number, line in enumerate(raw.splitlines(), 1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvidenceNormalizationError(
                f"invalid {label} row {number}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise EvidenceNormalizationError(f"{label} row {number} is not an object")
        rows.append(value)
    return rows


def _request_contract(receipt: Mapping[str, Any]) -> tuple[str, dict[str, str], dict[str, Any]]:
    block_id = receipt.get("block_id")
    generations = receipt.get("generations")
    if not isinstance(block_id, str) or not isinstance(generations, list) or len(generations) != 6:
        raise EvidenceNormalizationError("raw receipt lacks the exact six-request identity")
    roles: dict[str, str] = {}
    source_identity: dict[str, Any] | None = None
    for row in generations:
        if not isinstance(row, Mapping):
            raise EvidenceNormalizationError("raw generation row is not a mapping")
        request_id = row.get("request_id")
        role = row.get("role")
        driver = row.get("driver")
        if not isinstance(request_id, str) or not isinstance(role, str) or not isinstance(driver, Mapping):
            raise EvidenceNormalizationError("raw generation identity is incomplete")
        observed = driver.get("source_identity")
        if not isinstance(observed, Mapping) or not observed:
            raise EvidenceNormalizationError("driver source identity is missing")
        frozen = dict(observed)
        if source_identity is None:
            source_identity = frozen
        elif source_identity != frozen:
            raise EvidenceNormalizationError("driver source identity drift between requests")
        roles[request_id] = role
    if len(roles) != 6 or source_identity is None:
        raise EvidenceNormalizationError("request identities are not unique")
    if tuple(roles.values()) != _EXPECTED_ROLES:
        raise EvidenceNormalizationError("raw receipt request role order drift")
    return block_id, roles, source_identity


def _validate_bindings(runtime: Mapping[str, Any], round13: Mapping[str, Any]) -> None:
    if (
        runtime.get("status") != "ROUND16_A1_OBSERVER_BOUND"
        or runtime.get("host_layers") != list(HOST_LAYERS)
        or runtime.get("device_sample_layers") != list(DEVICE_SAMPLE_LAYERS)
        or runtime.get("device_query") != "DEFERRED_QUERY_ONLY_NO_SYNCHRONIZE"
    ):
        raise EvidenceNormalizationError("A1 runtime binding coverage is incomplete")
    if (
        round13.get("status") != "ROUND13_P0_ATTRIBUTION_BOUND"
        or round13.get("bindings") != _ROUND13_BINDINGS
        or round13.get("device_query") != "DEFERRED_QUERY_ONLY_NO_SYNCHRONIZE"
    ):
        raise EvidenceNormalizationError("frozen Round13 attribution binding drift")


def normalize_observer_rows(
    receipt: Mapping[str, Any],
    runtime_rows: Sequence[Mapping[str, Any]],
    round13_batches: Sequence[Mapping[str, Any]],
    *,
    runtime_bindings: Mapping[str, Any],
    round13_bindings: Mapping[str, Any],
) -> dict[str, Any]:
    """Return analyzer-ready rows plus an explicit conservative disposition."""
    _validate_bindings(runtime_bindings, round13_bindings)
    block_id, roles, source_identity = _request_contract(receipt)
    rows = [
        dict(row)
        for row in runtime_rows
        if row.get("component") not in {"NPU_RESIDENT", "NPU_SHARED"}
    ]
    if not rows:
        raise EvidenceNormalizationError("A1 runtime observer rows are empty")
    if any(row.get("block_id") != block_id for row in rows):
        raise EvidenceNormalizationError("A1 runtime row block identity drift")
    for request_id in roles:
        for component in _HOST_COMPONENTS:
            observed_layers = {
                row.get("layer")
                for row in rows
                if row.get("request_id") == request_id
                and row.get("component") == component
            }
            if observed_layers != set(HOST_LAYERS):
                raise EvidenceNormalizationError(
                    f"{component} Host coverage is not exactly layers 0..47 for {request_id}"
                )

    device_seen: dict[tuple[str, str], set[int]] = {}
    decode_batches: dict[str, list[Mapping[str, Any]]] = {item: [] for item in roles}
    serialized_request_order: list[str] = []
    serial = 0
    for batch in round13_batches:
        if (
            not isinstance(batch, Mapping)
            or batch.get("status") != "ROUND13_P0_BATCH"
            or batch.get("request_id") not in roles
            or batch.get("dropped_spans") != 0
            or batch.get("observer_errors") != []
            or not isinstance(batch.get("spans"), list)
        ):
            raise EvidenceNormalizationError("Round13 batch identity or completeness drift")
        request_id = str(batch["request_id"])
        serialized_request_order.append(request_id)
        batch_sequence = batch.get("batch_sequence")
        if type(batch_sequence) is not int or batch_sequence < 0:
            raise EvidenceNormalizationError("Round13 batch sequence is invalid")
        if batch.get("mode") == "decode":
            started = batch.get("started_ns")
            ended = batch.get("ended_ns")
            duration = batch.get("duration_ns")
            if (
                type(started) is not int
                or type(ended) is not int
                or type(duration) is not int
                or started < 0
                or ended < started
                or ended - started != duration
            ):
                raise EvidenceNormalizationError("Round13 Decode batch wall is invalid")
            decode_batches[request_id].append(batch)
        for index, span in enumerate(batch["spans"]):
            if not isinstance(span, Mapping) or span.get("component") not in {
                "QSA_INDEXER",
                "GDN",
                "NPU_RESIDENT",
            }:
                continue
            component = str(span["component"])
            layer = span.get("layer")
            expected_layers = (
                QSA_LAYERS
                if component == "QSA_INDEXER"
                else GDN_LAYERS
                if component == "GDN"
                else DEVICE_SAMPLE_LAYERS
            )
            elapsed = span.get("device_ms")
            if (
                layer not in expected_layers
                or span.get("device_status") != "COMPLETE"
                or span.get("observer_errors") != []
                or isinstance(elapsed, bool)
                or not isinstance(elapsed, (int, float))
                or not math.isfinite(float(elapsed))
                or float(elapsed) < 0
            ):
                raise EvidenceNormalizationError(
                    f"Round13 {component} direct event is incomplete"
                )
            serial += 1
            stem = f"{block_id}:{request_id}:round13:b{batch_sequence}:{component}:l{layer}:i{index}:n{serial}"
            parent_span_id = f"{block_id}:{request_id}:round13:b{batch_sequence}"
            correlation_id = parent_span_id
            if component == "NPU_RESIDENT":
                parents = [
                    row
                    for row in rows
                    if row.get("request_id") == request_id
                    and row.get("component") == "PARENT_ROUTED_EXPERT"
                    and row.get("layer") == layer
                ]
                if not parents:
                    raise EvidenceNormalizationError(
                        f"Round13 NPU_RESIDENT lacks A1 Host parent for layer {layer}"
                    )
                parent_span_id = str(parents[0]["span_id"])
                correlation_id = str(parents[0]["correlation_id"])
            rows.append(
                {
                    "schema": 1,
                    "block_id": block_id,
                    "request_id": request_id,
                    "component": component,
                    "layer": layer,
                    "span_id": stem,
                    "parent_span_id": parent_span_id,
                    "stream_id": "round13-npu-event-stream",
                    "start_event_id": f"{stem}:start",
                    "end_event_id": f"{stem}:end",
                    "correlation_id": correlation_id,
                    "clock_domain": "NPU_EVENT_ELAPSED_MS",
                    "start_ns": None,
                    "end_ns": None,
                    "duration_ns": None,
                    "elapsed_ms": float(elapsed),
                    "status": "COMPLETE",
                    "query_policy": "DEFERRED_QUERY_NO_SYNCHRONIZE",
                    "derivation": "DIRECT_OBSERVATION",
                }
            )
            device_seen.setdefault((request_id, component), set()).add(int(layer))

    for request_id in roles:
        if roles[request_id] not in {"W1_MEASUREMENT", "W2_MEASUREMENT"}:
            continue
        if device_seen.get((request_id, "QSA_INDEXER")) != set(QSA_LAYERS):
            raise EvidenceNormalizationError(
                f"measurement QSA_INDEXER coverage missing for {request_id}"
            )
        if device_seen.get((request_id, "GDN")) != set(GDN_LAYERS):
            raise EvidenceNormalizationError(
                f"measurement GDN coverage missing for {request_id}"
            )
        if device_seen.get((request_id, "NPU_RESIDENT")) != set(DEVICE_SAMPLE_LAYERS):
            raise EvidenceNormalizationError(
                f"measurement Round13 NPU_RESIDENT coverage missing for {request_id}"
            )
        batches = decode_batches[request_id]
        if not batches:
            raise EvidenceNormalizationError(f"Decode wall missing for {request_id}")
        started = min(int(batch["started_ns"]) for batch in batches)
        ended = max(int(batch["ended_ns"]) for batch in batches)
        serial += 1
        stem = f"{block_id}:{request_id}:decode-wall:n{serial}"
        rows.append(
            {
                "schema": 1,
                "block_id": block_id,
                "request_id": request_id,
                "component": "DECODE_PHASE_WALL",
                "layer": -1,
                "span_id": stem,
                "parent_span_id": f"{block_id}:{request_id}:scheduler",
                "stream_id": "host-scheduler-serial",
                "start_event_id": f"{stem}:start",
                "end_event_id": f"{stem}:end",
                "correlation_id": f"{block_id}:{request_id}:decode-phase",
                "clock_domain": "HOST_MONOTONIC_NS",
                "start_ns": started,
                "end_ns": ended,
                "duration_ns": ended - started,
                "elapsed_ms": None,
                "status": "COMPLETE",
                "derivation": "DIRECT_SERIAL_SCHEDULER_BOUNDING_INTERVAL",
            }
        )

    measurement_request_ids = sorted(
        request_id
        for request_id, role in roles.items()
        if role in {"W1_MEASUREMENT", "W2_MEASUREMENT"}
    )
    short_drain_request_id = next(
        request_id for request_id, role in roles.items() if role == "SHORT_DRAIN"
    )
    w2_request_id = next(
        request_id for request_id, role in roles.items() if role == "W2_MEASUREMENT"
    )
    if short_drain_request_id not in serialized_request_order:
        raise EvidenceNormalizationError(
            "SHORT_DRAIN serialized batch is required as the measurement flush witness"
        )
    w2_positions = [
        index
        for index, request_id in enumerate(serialized_request_order)
        if request_id == w2_request_id
    ]
    drain_positions = [
        index
        for index, request_id in enumerate(serialized_request_order)
        if request_id == short_drain_request_id
    ]
    if not w2_positions or min(drain_positions) <= max(w2_positions):
        raise EvidenceNormalizationError(
            "SHORT_DRAIN serialized batch is not a post-W2 measurement flush witness"
        )
    return {
        "schema": 1,
        "source_identity": source_identity,
        "rows": rows,
        "synchronize_inserted": False,
        "query_policy": "DEFERRED_QUERY_NO_SYNCHRONIZE",
        "aggregations": [],
        "candidate_regions": [],
        "measurement_event_flush": {
            "status": "COMPLETE_MEASUREMENT_BATCHES_SERIALIZED",
            "request_ids": measurement_request_ids,
            "required_direct_device_components": [
                "QSA_INDEXER",
                "GDN",
                "NPU_RESIDENT",
            ],
            "completion_basis": (
                "all required measurement events are COMPLETE in serialized "
                "Round13 batches before analysis"
            ),
            "trailing_schedule_roles": ["SHORT_DRAIN", "FIXED_CANARY"],
            "serialized_witness_role": "SHORT_DRAIN",
            "serialized_witness_request_id": short_drain_request_id,
            "final_canary_device_completion_required": False,
        },
        "attribution_status": STATUS_INSUFFICIENT,
        "attribution_reason": (
            "Round13 and A1 provide direct provider spans and a serial scheduler Decode "
            "bounding interval, but no direct isolated HOST_SYNC_EXPOSED_WAIT or "
            "HOST_LAUNCH_METADATA_CRITICAL interval; nested spans are not added or subtracted"
        ),
    }


def write_analysis_input(
    *,
    receipt_path: str | Path,
    runtime_rows_path: str | Path,
    runtime_bindings_path: str | Path,
    round13_rows_path: str | Path,
    round13_bindings_path: str | Path,
    output_rows_path: str | Path,
    output_metadata_path: str | Path,
) -> dict[str, Any]:
    receipt_file = Path(receipt_path).resolve()
    rows_file = Path(output_rows_path).resolve()
    metadata_file = Path(output_metadata_path).resolve()
    receipt = _json(receipt_file, "raw receipt")
    expected_rows = receipt.get("observer_path")
    if not isinstance(expected_rows, str) or Path(expected_rows).resolve() != rows_file:
        raise EvidenceNormalizationError("raw receipt observer path differs from normalized rows")
    payload = normalize_observer_rows(
        receipt,
        _jsonl(Path(runtime_rows_path), "A1 runtime observer JSONL"),
        _jsonl(Path(round13_rows_path), "Round13 attribution JSONL"),
        runtime_bindings=_json(Path(runtime_bindings_path), "A1 runtime bindings"),
        round13_bindings=_json(Path(round13_bindings_path), "Round13 bindings"),
    )
    if rows_file.exists() or metadata_file.exists():
        raise EvidenceNormalizationError("fresh normalized evidence outputs are required")
    with rows_file.open("x", encoding="utf-8") as destination:
        for row in payload["rows"]:
            destination.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    metadata = {
        key: payload[key]
        for key in (
            "schema",
            "source_identity",
            "synchronize_inserted",
            "query_policy",
            "aggregations",
            "candidate_regions",
            "measurement_event_flush",
            "attribution_status",
            "attribution_reason",
        )
    }
    metadata.update(
        receipt_sha256=_sha256(receipt_file),
        rows_path=str(rows_file),
        rows_sha256=_sha256(rows_file),
        normalization="ROUND13_V295_QSA_GDN_NPU_DIRECT_EVENTS_PLUS_A1_HOST48_NO_NESTED_ARITHMETIC",
    )
    metadata_file.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    return metadata


__all__ = (
    "DEVICE_SAMPLE_LAYERS",
    "EvidenceNormalizationError",
    "GDN_LAYERS",
    "HOST_LAYERS",
    "QSA_LAYERS",
    "normalize_observer_rows",
    "write_analysis_input",
)
