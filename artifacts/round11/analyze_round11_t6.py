#!/usr/bin/env python3
"""Strictly validate and compare the four unprofiled Round11 T6 blocks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import tarfile


EXPECTED_BLOCKS = {
    147: ("BASELINE", "A", 1),
    148: ("FINAL", "A", 0),
    149: ("FINAL", "B", 1),
    150: ("BASELINE", "B", 0),
}
EXPECTED_CLEANUP = {
    "guard_exit_ack": True,
    "container_removed": True,
    "physical12_after": "EMPTY",
}
PERFORMANCE_DECODE_LENGTHS = (32, 64, 128)
CANARY_CLASS_BY_CASE = {
    "canary_math391": "arithmetic",
    "canary_marker": "marker",
    "canary_json": "json",
    "canary_quote": "quote",
    "canary_transform": "transform",
}


def _finite_positive(value, label):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"invalid {label}")
    return float(value)


def _finite_nonnegative(value, label):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"invalid {label}")
    return float(value)


def _validate_path(row):
    path = row.get("path")
    if (not isinstance(path, dict) or path.get("path") != "HYBRID"
            or type(path.get("hybrid_provider_event_count")) is not int
            or path["hybrid_provider_event_count"] <= 0
            or not isinstance(path.get("chunks"), list) or not path["chunks"]):
        raise ValueError("Hybrid path contract failed")


def _output_hash(output_ids):
    payload = json.dumps(output_ids, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _validate_document(document):
    version = document.get("version")
    if version not in EXPECTED_BLOCKS:
        raise ValueError("unexpected T6 version")
    mode, block, aborts = EXPECTED_BLOCKS[version]
    if document.get("mode") != mode or document.get("block") != block:
        raise ValueError("T6 block identity mismatch")
    if (document.get("status") != "PASS_ROUND11_T6_PROCESS"
            or document.get("profiler") != "OFF"
            or not isinstance(document.get("runtime_identity"), str)
            or not document["runtime_identity"]
            or not isinstance(document.get("fixture_sha256"), str)
            or not document["fixture_sha256"]):
        raise ValueError("T6 process identity mismatch")
    if (document.get("generation_budget") != 6
            or document.get("generation_attempts") != 6
            or document.get("automatic_retries") != 0
            or document.get("abort_controls") != aborts):
        raise ValueError("T6 generation contract failed")
    if document.get("cleanup") != EXPECTED_CLEANUP:
        raise ValueError("T6 cleanup contract failed")
    receipts = document.get("cancel_receipts")
    if (not isinstance(receipts, list) or len(receipts) != aborts
            or any(value != "round11-t6-cancel-receipt.json" for value in receipts)):
        raise ValueError("T6 cancel receipt contract failed")
    rows = document.get("results")
    if not isinstance(rows, list) or len(rows) != 6:
        raise ValueError("T6 result count mismatch")
    if [row.get("index") for row in rows] != list(range(1, 7)):
        raise ValueError("T6 result sequence mismatch")

    warmup = rows[0]
    if (warmup.get("case_id") != "canary_math391"
            or warmup.get("phase") != "warmup"
            or warmup.get("quality_pass") is not True):
        raise ValueError("T6 warmup quality contract failed")
    _finite_positive(warmup.get("e2e_seconds"), "warmup E2E")
    _validate_path(warmup)

    performance = {}
    for row, decode_tokens in zip(rows[1:4], PERFORMANCE_DECODE_LENGTHS):
        if (row.get("case_id") != "edge128"
                or row.get("phase") != "performance"
                or row.get("input_tokens") != 128
                or row.get("decode_tokens") != decode_tokens
                or row.get("completion_tokens") != decode_tokens
                or row.get("quality_pass") is not True
                or row.get("finite_logprobs") is not True
                or row.get("finish_reason") != "length"
                or row.get("saw_done_marker") is not True):
            raise ValueError("T6 performance quality contract failed")
        _finite_positive(row.get("e2e_seconds"), "performance E2E")
        _validate_path(row)
        output_ids = row.get("output_ids")
        arrivals = row.get("token_arrival_monotonic_ns")
        if (not isinstance(output_ids, list) or len(output_ids) != decode_tokens
                or not all(type(token) is int for token in output_ids)
                or not isinstance(arrivals, list) or len(arrivals) != decode_tokens
                or not all(type(value) is int for value in arrivals)
                or any(right <= left for left, right in zip(arrivals, arrivals[1:]))):
            raise ValueError("T6 streamed output contract failed")
        gaps = [(right - left) / 1e9 for left, right in zip(arrivals, arrivals[1:])]
        performance[decode_tokens] = {
            "e2e_seconds": float(row["e2e_seconds"]),
            "output_hash": _output_hash(output_ids),
            "output_ids": output_ids,
            "median_inter_token_seconds": statistics.median(gaps) if gaps else None,
        }

    if aborts:
        cancel, reuse = rows[4:]
        if (cancel.get("case_id") != "cancel128" or cancel.get("phase") != "cancel"
                or cancel.get("cancel_status")
                != "PASS_HYBRID_MIDPREFILL_CANCEL_AND_REUSE"):
            raise ValueError("T6 cancellation contract failed")
        _finite_positive(cancel.get("e2e_seconds"), "cancel E2E")
        if (reuse.get("case_id") != "canary_math391" or reuse.get("phase") != "reuse"
                or reuse.get("quality_pass") is not True):
            raise ValueError("T6 reuse contract failed")
        _finite_positive(reuse.get("e2e_seconds"), "reuse E2E")
        _validate_path(reuse)
    else:
        expected_extras = (("canary_marker", "canary_json") if version == 148
                           else ("canary_quote", "canary_transform"))
        for row, case_id in zip(rows[4:], expected_extras):
            if (row.get("case_id") != case_id or row.get("phase") != "canary"
                    or row.get("quality_pass") is not True):
                raise ValueError("T6 canary coverage mismatch")
            _finite_positive(row.get("e2e_seconds"), "canary E2E")
            _validate_path(row)
    return performance


def _resource_summary(rows):
    if not isinstance(rows, list) or not rows:
        raise ValueError("T6 resource monitor evidence missing")
    hbm = []
    owned = []
    host_available = []
    cgroup_current = []
    cgroup_peak = []
    numa_free = {str(node): [] for node in (2, 3, 4, 7)}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("invalid T6 resource monitor row")
        if "action" in row and "hbm_used_mib" not in row:
            continue
        hbm.append(_finite_nonnegative(row.get("hbm_used_mib"), "HBM usage"))
        owned.append(_finite_nonnegative(row.get("owned_process_mib"), "owned HBM usage"))
        host_available.append(_finite_positive(row.get("host_mem_available_kib"),
                                                "host available memory"))
        try:
            for node in numa_free:
                numa_free[node].append(int(row["numa_meminfo_kib"][node]["MemFree"]))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("incomplete T6 resource monitor row") from error
        cgroup = row.get("owned_cgroup")
        if cgroup:
            try:
                cgroup_current.append(int(cgroup["memory.usage_in_bytes"]))
                cgroup_peak.append(int(cgroup["memory.max_usage_in_bytes"]))
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("incomplete T6 cgroup monitor row") from error
    if not hbm or not cgroup_current or any(
            value <= 0 for value in cgroup_current + cgroup_peak
            + [value for values in numa_free.values() for value in values]):
        raise ValueError("non-positive T6 resource monitor value")
    return {
        "max_hbm_used_mib": max(hbm),
        "max_owned_process_mib": max(owned),
        "max_host_cgroup_current_gib": max(cgroup_current) / 2**30,
        "max_host_cgroup_peak_gib": max(cgroup_peak) / 2**30,
        "min_host_available_gib": min(host_available) / 2**20,
        "min_numa_memfree_gib": {
            node: min(values) / 2**20 for node, values in numa_free.items()
        },
    }


def analyze_t6(documents, monitors):
    if len(documents) != 4:
        raise ValueError("T6 requires four independent model-process blocks")
    by_version = {}
    performance = {}
    for document in documents:
        version = document.get("version")
        if version in by_version:
            raise ValueError("duplicate T6 version")
        by_version[version] = document
        performance[version] = _validate_document(document)
    if set(by_version) != set(EXPECTED_BLOCKS):
        raise ValueError("T6 ABBA version coverage mismatch")
    identities = {document["runtime_identity"] for document in documents}
    fixtures = {document["fixture_sha256"] for document in documents}
    if len(identities) != 1 or len(fixtures) != 1:
        raise ValueError("T6 cross-block identity mismatch")

    for decode_tokens in PERFORMANCE_DECODE_LENGTHS:
        hashes = {performance[version][decode_tokens]["output_hash"]
                  for version in EXPECTED_BLOCKS}
        ids = {tuple(performance[version][decode_tokens]["output_ids"])
               for version in EXPECTED_BLOCKS}
        if len(hashes) != 1 or len(ids) != 1:
            raise ValueError("T6 performance output identity mismatch")

    comparison = {}
    for decode_tokens in PERFORMANCE_DECODE_LENGTHS:
        baseline = [performance[version][decode_tokens]["e2e_seconds"]
                    for version in (147, 150)]
        final = [performance[version][decode_tokens]["e2e_seconds"]
                 for version in (148, 149)]
        baseline_median = statistics.median(baseline)
        final_median = statistics.median(final)
        comparison[str(decode_tokens)] = {
            "baseline_blocks_seconds": baseline,
            "final_blocks_seconds": final,
            "baseline_median_seconds": baseline_median,
            "final_median_seconds": final_median,
            "final_over_baseline": final_median / baseline_median,
            "output_sha256": performance[147][decode_tokens]["output_hash"],
            "per_block_median_inter_token_seconds": {
                str(version): performance[version][decode_tokens][
                    "median_inter_token_seconds"]
                for version in EXPECTED_BLOCKS
            },
        }

    if set(monitors) != set(EXPECTED_BLOCKS):
        raise ValueError("T6 resource block coverage mismatch")
    resources = {
        str(version): _resource_summary(monitors[version])
        for version in EXPECTED_BLOCKS
    }
    all_cases = {row["case_id"] for document in documents for row in document["results"]}
    canary_classes = sorted(CANARY_CLASS_BY_CASE[case_id]
                            for case_id in CANARY_CLASS_BY_CASE if case_id in all_cases)
    if canary_classes != ["arithmetic", "json", "marker", "quote", "transform"]:
        raise ValueError("T6 five-class canary coverage incomplete")

    return {
        "schema": 1,
        "status": "PASS_ROUND11_T6_FINAL_INTEGRATION",
        "disposition": "HFINAL_REQUALIFIED_NO_NEW_PROMOTION",
        "runtime_identity": identities.pop(),
        "fixture_sha256": fixtures.pop(),
        "block_order": [
            {"version": version, "mode": EXPECTED_BLOCKS[version][0],
             "block": EXPECTED_BLOCKS[version][1]}
            for version in EXPECTED_BLOCKS
        ],
        "quality": {
            "five_deterministic_canary_classes": True,
            "canary_classes": canary_classes,
            "input_tokens": 128,
            "decode_lengths": list(PERFORMANCE_DECODE_LENGTHS),
            "all_performance_output_ids_exact_across_blocks": True,
            "all_performance_logprobs_finite": True,
            "all_performance_paths": "HYBRID",
            "abort_controls": sum(document["abort_controls"] for document in documents),
            "same_process_reuse_controls": 2,
            "cleanup": "ALL_GUARD_ACK_CONTAINER_REMOVED_PHYSICAL12_EMPTY",
        },
        "performance": {
            "profiler": "OFF",
            "same_compute_identity": True,
            "comparison": comparison,
            "interpretation": (
                "BASELINE and FINAL labels execute the same frozen HFINAL compute identity. "
                "Ratios quantify independent-process reproducibility and process variance; "
                "they are not a candidate speedup claim."
            ),
        },
        "resources": {"blocks": resources},
        "budget": {
            "model_processes": 4,
            "generation_attempts": 24,
            "abort_controls": 2,
            "automatic_retries": 0,
        },
        "limitations": [
            "HFINAL contains no newly promoted Kunpeng or B-route candidate.",
            "T6 records unprofiled E2E and streamed inter-token arrival gaps; it does not infer an independent TTFT or prefill duration.",
            "Qualification remains serial TP1 and the Round11 tested Hybrid boundary; it is not production deployment qualification.",
        ],
    }


def _read_archive(path):
    with tarfile.open(path, "r:gz") as archive:
        names = archive.getnames()
        finals = [name for name in names if name.endswith("/round11-t6-final.json")]
        monitors = [name for name in names if name.endswith("/monitor.jsonl")]
        if len(finals) != 1 or len(monitors) != 1:
            raise ValueError(f"invalid T6 archive contents: {path}")
        document = json.load(archive.extractfile(finals[0]))
        rows = []
        for line in archive.extractfile(monitors[0]):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return document, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", action="append", type=Path, required=True,
                        help="T6 evidence archive; pass exactly four times")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    documents = []
    monitors = {}
    for path in args.archive:
        document, rows = _read_archive(path)
        documents.append(document)
        monitors[document["version"]] = rows
    result = analyze_t6(documents, monitors)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(result["status"], result["disposition"])


if __name__ == "__main__":
    main()
