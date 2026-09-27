#!/usr/bin/env python3
"""Validate and summarize the two formal Round14 H512 qualification blocks."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics


EXPECTED = ((269, "H512", "A"), (270, "H512", "B"))
ALLOWED_PROCESS_IDENTITIES = set(EXPECTED) | {(273, "H512", "B")}
BUCKETS = ("LE128", "129_256", "257_512")
EXPECTED_COUNTS = {
    "quality": 48,
    "performance": 12,
    "boundary_accept": 1,
    "boundary_reject": 1,
    "cancel": 1,
    "reuse": 1,
}
EXPECTED_CLEANUP = {
    "guard_exit_ack": True,
    "container_removed": True,
    "physical12_after": "EMPTY",
}
EXPECTED_LIMITS = {
    "max_input_tokens": 512,
    "max_output_tokens": 256,
    "max_total_tokens": 768,
    "hybrid_threshold_tokens": 513,
}


def _finite_positive(value, label):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise RuntimeError(f"invalid {label}")
    return float(value)


def _finite_logprobs(row):
    """Accept the two frozen driver field names without hiding conflicts."""
    values = [
        row[name]
        for name in ("finite_logprobs", "finite_output_logprobs")
        if name in row
    ]
    return bool(values) and all(value is True for value in values)


def _read_jsonl(path):
    raw = Path(path).read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise RuntimeError("unterminated Round14 B monitor JSONL")
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def summarize_resources(rows):
    samples = [row for row in rows if "host_mem_available_kib" in row]
    if not samples:
        raise RuntimeError("Round14 B monitor has no resource samples")
    hbm = [int(row["hbm_used_mib"]) for row in samples if "hbm_used_mib" in row]
    if not hbm:
        raise RuntimeError("Round14 B monitor has no HBM samples")
    owned = [int(row.get("owned_process_mib", 0)) for row in samples]
    cgroup = []
    for row in samples:
        values = row.get("owned_cgroup", {})
        for name in (
            "memory.peak", "memory.max_usage_in_bytes", "memory.current",
            "memory.usage_in_bytes",
        ):
            if values.get(name) is not None:
                cgroup.append(int(values[name]))
    numa = {}
    for node in ("2", "3", "4", "7"):
        values = [
            int(row["numa_meminfo_kib"][node]["MemFree"])
            for row in samples if node in row.get("numa_meminfo_kib", {})
        ]
        numa[node] = min(values) if values else None
    return {
        "physical_device": 12,
        "whole_card_hbm_peak_mib": max(hbm),
        "owned_process_hbm_peak_mib": max(owned) if owned else None,
        "host_cgroup_peak_bytes": max(cgroup) if cgroup else None,
        "minimum_host_mem_available_kib": min(
            int(row["host_mem_available_kib"]) for row in samples
        ),
        "minimum_numa_memfree_kib": numa,
        "sample_count": len(samples),
    }


def _validate_provider(row):
    provider = row.get("provider")
    completion = row.get("completion_tokens")
    if not isinstance(provider, dict):
        raise RuntimeError("Round14 B provider contract missing")
    boundary = provider.get("boundary", {})
    qsa = provider.get("qsa", {})
    if (
        boundary.get("status") != "ROUND14_H_BOUNDED_ACCEPT"
        or boundary.get("before_scheduler_dispatch") is not True
        or not isinstance(provider.get("hybrid"), dict)
        or qsa.get("prefill_strategy") != "VECTORIZED"
        or qsa.get("decode_strategy") != "DIRECT_BF16_ROW"
        or qsa.get("actual_completion_tokens") != completion
        or qsa.get("request_drained") is not True
        or not isinstance(provider.get("ple"), dict)
    ):
        raise RuntimeError("Round14 B provider contract failed")


def _validate_process(document):
    identity = (
        document.get("version"), document.get("profile"), document.get("block")
    )
    if (
        identity not in ALLOWED_PROCESS_IDENTITIES
        or document.get("status") != "PASS_ROUND14_B_MODEL_PROCESS"
        or document.get("stage") != "ROUND14_B_REAL_DISTRIBUTION_AND_BOUNDED_HYBRID"
        or document.get("limits") != EXPECTED_LIMITS
        or document.get("profiler") != "OFF"
        or document.get("generation_budget") != 64
        or document.get("generation_attempts") != 64
        or document.get("abort_controls") != 1
        or document.get("automatic_retries") != 0
        or document.get("cleanup") != EXPECTED_CLEANUP
        or document.get("cancel_receipts") != ["round14-b-cancel-receipt.json"]
        or document.get("state_pool_matrix", {}).get("status")
            != "PASS_ACTUAL_CONTEXT_POOL_GEOMETRY"
    ):
        raise RuntimeError("Round14 B process contract failed")
    results = document.get("results")
    if not isinstance(results, list) or len(results) != 64:
        raise RuntimeError("Round14 B process contract failed: result count")
    if [row.get("ordinal") for row in results] != list(range(1, 65)):
        raise RuntimeError("Round14 B process contract failed: ordinal sequence")
    counts = {
        kind: sum(row.get("kind") == kind for row in results)
        for kind in EXPECTED_COUNTS
    }
    if counts != EXPECTED_COUNTS:
        raise RuntimeError("Round14 B schedule coverage mismatch")

    quality = [row for row in results if row["kind"] == "quality"]
    performance = [row for row in results if row["kind"] == "performance"]
    for bucket in BUCKETS:
        selected_quality = [row for row in quality if row.get("bucket") == bucket]
        selected_performance = [row for row in performance if row.get("bucket") == bucket]
        if len(selected_quality) != 16 or len(selected_performance) != 4:
            raise RuntimeError("Round14 B bucket coverage mismatch")
        if sum(row.get("split") == "HOLDOUT" for row in selected_quality) != 12:
            raise RuntimeError("Round14 B HOLDOUT coverage mismatch")
    for row in quality:
        if (
            row.get("quality_status")
                not in {"PASS_AUTOMATIC", "FAIL_AUTOMATIC", "PENDING_HUMAN"}
            or not _finite_logprobs(row)
        ):
            raise RuntimeError("Round14 B quality evidence malformed")
        _finite_positive(row.get("e2e_seconds"), "quality E2E")
        _validate_provider(row)
    for row in performance:
        if not _finite_logprobs(row) or row.get("finish_reason") != "length":
            raise RuntimeError("Round14 B fixed-length performance output failed")
        for name in ("e2e_seconds", "ttft_seconds", "tpot_seconds"):
            _finite_positive(row.get(name), name)
        _validate_provider(row)

    accept = next(row for row in results if row["kind"] == "boundary_accept")
    if (
        accept.get("input_tokens") != 512
        or accept.get("completion_tokens") != 256
        or accept.get("finish_reason") != "length"
        or not _finite_logprobs(accept)
    ):
        raise RuntimeError("Round14 B accepted boundary contract failed")
    _validate_provider(accept)
    reject = next(row for row in results if row["kind"] == "boundary_reject")
    if (
        reject.get("expected_rejection") is not True
        or not 400 <= reject.get("http_status", 0) < 500
        or reject.get("boundary", {}).get("status") != "ROUND14_H_BOUNDED_REJECT"
        or reject.get("boundary", {}).get("before_scheduler_dispatch") is not True
    ):
        raise RuntimeError("Round14 B rejected boundary contract failed")
    cancel = next(row for row in results if row["kind"] == "cancel")
    reuse = next(row for row in results if row["kind"] == "reuse")
    if (
        not cancel.get("cancel_status")
        or cancel.get("boundary", {}).get("status") != "ROUND14_H_BOUNDED_ACCEPT"
        or reuse.get("quality_pass") is not True
        or reuse.get("quality_status") != "PASS_AUTOMATIC"
        or not _finite_logprobs(reuse)
    ):
        raise RuntimeError("Round14 B cancel/reuse contract failed")
    _validate_provider(reuse)
    return results


def _median(rows, name):
    return statistics.median(_finite_positive(row.get(name), name) for row in rows)


def _summarize_process(document, resources):
    results = _validate_process(document)
    quality = [row for row in results if row["kind"] == "quality"]
    performance = [row for row in results if row["kind"] == "performance"]
    quality_summary = {
        "requests": len(quality),
        "dev": sum(row.get("split") == "DEV" for row in quality),
        "holdout": sum(row.get("split") == "HOLDOUT" for row in quality),
        "automatic_pass": sum(row.get("quality_status") == "PASS_AUTOMATIC" for row in quality),
        "automatic_fail": sum(row.get("quality_status") == "FAIL_AUTOMATIC" for row in quality),
        "human_review_pending": sum(row.get("quality_status") == "PENDING_HUMAN" for row in quality),
        "by_bucket": {},
        "by_category": {},
    }
    for bucket in BUCKETS:
        selected = [row for row in quality if row.get("bucket") == bucket]
        quality_summary["by_bucket"][bucket] = {
            "requests": len(selected),
            "automatic_pass": sum(row.get("quality_status") == "PASS_AUTOMATIC" for row in selected),
            "automatic_fail": sum(row.get("quality_status") == "FAIL_AUTOMATIC" for row in selected),
            "human_review_pending": sum(row.get("quality_status") == "PENDING_HUMAN" for row in selected),
            "e2e_median_seconds": _median(selected, "e2e_seconds"),
        }
    categories = sorted({row.get("category") for row in quality})
    for category in categories:
        selected = [row for row in quality if row.get("category") == category]
        quality_summary["by_category"][category] = {
            "requests": len(selected),
            "automatic_fail": sum(row.get("quality_status") == "FAIL_AUTOMATIC" for row in selected),
            "human_review_pending": sum(row.get("quality_status") == "PENDING_HUMAN" for row in selected),
        }
    performance_summary = {}
    for bucket in BUCKETS:
        selected = [row for row in performance if row.get("bucket") == bucket]
        performance_summary[bucket] = {
            "requests": len(selected),
            "actual_output_tokens": sum(int(row["completion_tokens"]) for row in selected),
            "e2e_median_seconds": _median(selected, "e2e_seconds"),
            "ttft_median_seconds": _median(selected, "ttft_seconds"),
            "tpot_median_seconds": _median(selected, "tpot_seconds"),
        }
    accept = next(row for row in results if row["kind"] == "boundary_accept")
    return {
        "version": document["version"],
        "block": document["block"],
        "quality": quality_summary,
        "performance_by_bucket": performance_summary,
        "boundary": {
            "accepted_input_tokens": accept["input_tokens"],
            "accepted_output_tokens": accept["completion_tokens"],
            "accepted_total_context": accept["input_tokens"] + accept["completion_tokens"],
            "outside_request_rejected_before_scheduler": True,
        },
        "cancel_reuse": "PASS",
        "provider": "VECTORIZED_PREFILL_DIRECT_BF16_ROW_DECODE",
        "resources": resources,
    }


def compare_documents(documents, resources):
    if len(documents) != 2 or len(resources) != 2:
        raise ValueError("Round14 B comparator requires two formal H512 blocks")
    observed = [
        (row.get("version"), row.get("profile"), row.get("block"))
        for row in documents
    ]
    if tuple(observed) != EXPECTED:
        raise RuntimeError(f"Round14 B process identity/order mismatch: {observed}")
    if len({row.get("fixture_sha256") for row in documents}) != 1:
        raise RuntimeError("Round14 B public proxy fixture identity drift")
    if len({row.get("edge_fixture_sha256") for row in documents}) != 1:
        raise RuntimeError("Round14 B edge fixture identity drift")
    summaries = [
        _summarize_process(document, resource)
        for document, resource in zip(documents, resources)
    ]
    blocks = {row["block"]: row for row in summaries}
    auto_fail = sum(row["quality"]["automatic_fail"] for row in summaries)
    pending = sum(row["quality"]["human_review_pending"] for row in summaries)
    if auto_fail:
        disposition = "NOT_DEFAULT_QUALITY_REGRESSION"
        default = "S0_LEGACY_H129"
    elif pending:
        disposition = "NOT_DEFAULT_PENDING_HUMAN_REVIEW"
        default = "S0_LEGACY_H129"
    else:
        disposition = "H512_DEFAULT_CANDIDATE_QUALIFIED"
        default = "H_BOUNDED_H512"
    return {
        "schema": 1,
        "status": "PASS_ROUND14_B_EVIDENCE_CLOSED",
        "process_order": [row["version"] for row in summaries],
        "fixture_sha256": documents[0]["fixture_sha256"],
        "fixture_scope": "PUBLIC_PROXY_DISTRIBUTION_NOT_BUSINESS_LOGS",
        "profile": "H512",
        "default_profile": default,
        "h512_disposition": disposition,
        "h768_disposition": (
            "NOT_RUN_CONDITION_NOT_MET_AND_B_MODEL_GROUP_CAP"
            if default == "S0_LEGACY_H129" else "NOT_RUN_B_MODEL_GROUP_CAP"
        ),
        "blocks": blocks,
        "coverage": {
            "frozen_fixture_requests": 64,
            "quality_requests_per_block": 48,
            "dev_quality_per_block": 12,
            "holdout_quality_per_block": 36,
            "tested_input_buckets": list(BUCKETS),
            "unexecuted_fixture_bucket": "513_768",
            "automatic_quality_failures_across_blocks": auto_fail,
            "human_review_pending_across_blocks": pending,
            "boundary_accept_input_output": [512, 256],
            "boundary_rejection_before_scheduler": True,
            "cancel_reuse_blocks": 2,
        },
        "claim_scope": (
            "H512 is an experimental forced-Hybrid qualification profile. It is not "
            "the default when automatic quality fails or human review is pending. "
            "The current default remains total tokens <129 Hybrid and >=129 legacy fallback."
        ),
    }


def close_partial(failed, supervisor, completed, resources):
    failed_results = failed.get("results")
    if (
        failed.get("status") != "FAIL_ROUND14_B_DRIVER"
        or (failed.get("version"), failed.get("profile"), failed.get("block"))
            != (269, "H512", "A")
        or failed.get("generation_attempts") != 61
        or failed.get("abort_controls") != 0
        or failed.get("automatic_retries") != 0
        or not isinstance(failed_results, list)
        or len(failed_results) != 60
        or [row.get("ordinal") for row in failed_results] != list(range(1, 61))
        or "stream did not produce the exact Decode token budget"
            not in failed.get("error", "")
    ):
        raise RuntimeError("Round14 B failed-block contract mismatch")
    if (
        supervisor.get("status") != "FAIL"
        or supervisor.get("version") != 269
        or supervisor.get("driver_exit") != 1
        or supervisor.get("owned_container_removed") is not True
        or supervisor.get("physical12_after") != "EMPTY"
        or supervisor.get("guard_exit_ack", {}).get("action") != "OWNED_EXITED"
    ):
        raise RuntimeError("Round14 B failed-block cleanup contract mismatch")
    if (
        (completed.get("version"), completed.get("profile"), completed.get("block"))
        != (273, "H512", "B")
        or failed.get("fixture_sha256") != completed.get("fixture_sha256")
        or failed.get("edge_fixture_sha256") != completed.get("edge_fixture_sha256")
    ):
        raise RuntimeError("Round14 B repair identity/fixture mismatch")
    failed_quality = [row for row in failed_results if row.get("kind") == "quality"]
    if len(failed_quality) != 48:
        raise RuntimeError("Round14 B failed block lacks complete quality distribution")
    for row in failed_quality:
        if (
            row.get("quality_status")
                not in {"PASS_AUTOMATIC", "FAIL_AUTOMATIC", "PENDING_HUMAN"}
            or not _finite_logprobs(row)
        ):
            raise RuntimeError("Round14 B failed-block quality evidence malformed")
        _validate_provider(row)
    complete_summary = _summarize_process(completed, resources)
    failed_failures = sum(
        row.get("quality_status") == "FAIL_AUTOMATIC" for row in failed_quality
    )
    failed_pending = sum(
        row.get("quality_status") == "PENDING_HUMAN" for row in failed_quality
    )
    complete_failures = complete_summary["quality"]["automatic_fail"]
    complete_pending = complete_summary["quality"]["human_review_pending"]
    if failed_failures or complete_failures:
        disposition = "NOT_DEFAULT_QUALITY_REGRESSION"
    elif failed_pending or complete_pending:
        disposition = "NOT_DEFAULT_PENDING_HUMAN_REVIEW_AND_INCOMPLETE_TWO_BLOCK_QUALIFICATION"
    else:
        disposition = "NOT_DEFAULT_INCOMPLETE_TWO_BLOCK_QUALIFICATION"
    return {
        "schema": 1,
        "status": "ROUND14_B_PARTIAL_WITH_ONE_COMPLETE_BLOCK",
        "process_order": [269, 273],
        "required_complete_blocks": 2,
        "complete_blocks": 1,
        "default_profile": "S0_LEGACY_H129",
        "h512_disposition": disposition,
        "h768_disposition": "NOT_RUN_CONDITION_NOT_MET_AND_B_MODEL_GROUP_CAP",
        "fixture_sha256": completed["fixture_sha256"],
        "fixture_scope": "PUBLIC_PROXY_DISTRIBUTION_NOT_BUSINESS_LOGS",
        "failed_block": {
            "version": 269,
            "block": "A",
            "completed_results": len(failed_results),
            "complete_quality_requests": len(failed_quality),
            "automatic_quality_fail": failed_failures,
            "human_review_pending": failed_pending,
            "failure": "HOST_ONLY_HYBRID_PATH_PROBE_4096_ROW_BOUND_EXHAUSTED_DURING_BOUNDARY_ACCEPT",
            "cleanup": "PASS",
            "usable_as_complete_confirmation_block": False,
        },
        "complete_block": complete_summary,
        "coverage": {
            "frozen_fixture_requests": 64,
            "complete_quality_distribution_blocks": 2,
            "complete_end_to_end_schedule_blocks": 1,
            "tested_input_buckets": list(BUCKETS),
            "unexecuted_fixture_bucket": "513_768",
            "boundary_cancel_reuse_complete_blocks": 1,
        },
        "claim_scope": (
            "H512 executed one complete 64-request process after the bounded host-only "
            "probe repair. The failed v269 prefix is retained but is not promoted to a "
            "second independent confirmation block. H129 remains the default."
        ),
    }


def compare(inputs):
    documents = [json.loads(Path(result).read_text()) for result, _monitor in inputs]
    resources = [summarize_resources(_read_jsonl(monitor)) for _result, monitor in inputs]
    return compare_documents(documents, resources)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a-final", type=Path, required=True)
    parser.add_argument("--a-monitor", type=Path, required=True)
    parser.add_argument("--b-final", type=Path, required=True)
    parser.add_argument("--b-monitor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = compare([
        (args.a_final, args.a_monitor),
        (args.b_final, args.b_monitor),
    ])
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(report["status"], report["h512_disposition"], flush=True)


if __name__ == "__main__":
    main()
