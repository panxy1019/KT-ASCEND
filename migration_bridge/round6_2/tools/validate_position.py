"""Fail-closed validation of Round6.2 position identity and Deferred telemetry.

Input rows are capture records, not numerical contract verdicts. Counts use
route events (one logical expert execution for one input token), not CPU tasks.
Prefix counts include all request tokens strictly BEFORE the current row.
"""
import math
import re


def validate(row):
    required = {
        "position_id", "request_id", "prompt_id", "prompt_hash", "history_hash",
        "request_input_hash", "token_index", "current_input_position", "forward_id", "logits_sha256",
        "logits_shape", "mode", "baseline_history_sha256", "trace_complete",
        "deferred_event_count_current_row", "deferred_event_count_prefix",
        "cpu_route_count_current_row", "cpu_route_count_prefix",
        "deferred_route_records", "deferred_exposure", "pending_drained",
    }
    if missing := required - row.keys():
        raise ValueError(f"MISSING_FIELDS: {sorted(missing)}")
    for key in ("prompt_hash", "history_hash", "request_input_hash", "logits_sha256", "baseline_history_sha256"):
        if not isinstance(row[key], str) or not re.fullmatch(r"[0-9a-f]{64}", row[key]):
            raise ValueError(f"INVALID_HASH: {key}")
    for key in ("position_id", "request_id", "prompt_id", "forward_id"):
        if not isinstance(row[key], str) or not row[key]:
            raise ValueError(f"INVALID_ID: {key}")
    if row["mode"] not in ("defer0", "defer1"):
        raise ValueError("INVALID_MODE")
    if row["trace_complete"] is not True or row["pending_drained"] is not True:
        raise ValueError("INCOMPLETE_TRACE_OR_PENDING")
    if row["logits_shape"] != [102400]:
        raise ValueError("INVALID_DSV2_LOGITS_SHAPE")
    counts = ("deferred_event_count_current_row", "deferred_event_count_prefix",
              "cpu_route_count_current_row", "cpu_route_count_prefix", "token_index", "current_input_position")
    for key in counts:
        if type(row[key]) is not int or row[key] < 0:
            raise ValueError(f"INVALID_COUNT: {key}")
    current, prefix = row[counts[0]], row[counts[1]]
    if current > row[counts[2]] or prefix > row[counts[3]]:
        raise ValueError("DEFERRED_COUNT_EXCEEDS_CPU")
    if row["mode"] == "defer0" and (current or prefix):
        raise ValueError("NODEFER_HAS_DEFERRED_EXPOSURE")
    expected = ("CURRENT_AND_PREFIX_EXPOSURE" if current and prefix else
                "CURRENT_EXPOSURE" if current else "PREFIX_ONLY_EXPOSURE" if prefix else "NO_EXPOSURE")
    if row["deferred_exposure"] != expected:
        raise ValueError("EXPOSURE_CLASS_MISMATCH")
    records = row["deferred_route_records"]
    if not isinstance(records, list) or len(records) != current + prefix:
        raise ValueError("MISSING_DEFERRED_ONLY_ROUTE_RECORDS")
    identities = set()
    for record in records:
        fields = {"event_id", "request_id", "producer_layer", "consumer_layer",
                  "expert_logical_id", "topk_rank", "token_position", "scope",
                  "route_score", "normalized_route_weight"}
        if fields - record.keys():
            raise ValueError("INCOMPLETE_ROUTE_RECORD")
        if record["event_id"] in identities or record["request_id"] != row["request_id"]:
            raise ValueError("DUPLICATE_OR_FOREIGN_ROUTE")
        identities.add(record["event_id"])
        if (record["producer_layer"], record["consumer_layer"]) != (7, 8):
            raise ValueError("NON_D1_DEFERRED_ROUTE")
        if record["expert_logical_id"] not in (16, 38, 41, 50):
            raise ValueError("NON_CPU_OWNED_ROUTE")
        if type(record["topk_rank"]) is not int or not 0 <= record["topk_rank"] < 6:
            raise ValueError("INVALID_TOPK_RANK")
        if type(record["token_position"]) is not int or record["token_position"] < 0:
            raise ValueError("INVALID_ROUTE_POSITION")
        if record["scope"] not in ("current", "prefix"):
            raise ValueError("INVALID_ROUTE_SCOPE")
        if ((record["scope"] == "current" and record["token_position"] != row["current_input_position"])
                or (record["scope"] == "prefix" and record["token_position"] >= row["current_input_position"])):
            raise ValueError("ROUTE_POSITION_SCOPE_MISMATCH")
        for key in ("route_score", "normalized_route_weight"):
            value = record[key]
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("INVALID_ROUTE_WEIGHT")
    if sum(r["scope"] == "current" for r in records) != current:
        raise ValueError("ROUTE_SCOPE_COUNT_MISMATCH")
    return expected
