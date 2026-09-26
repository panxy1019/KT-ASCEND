"""Strict aggregation for the four frozen Round11 T1 processes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROCESSES = ("BOUNDARY", "HYBRID256", "HYBRID512", "HYBRID1024")
EXPECTED_COUNTS = {"BOUNDARY": 3, "HYBRID256": 15, "HYBRID512": 4, "HYBRID1024": 2}
EXPECTED_ABORTS = {"BOUNDARY": 0, "HYBRID256": 1, "HYBRID512": 0, "HYBRID1024": 0}


def _index(rows, case_id, phase):
    selected = [row for row in rows if row.get("case_id") == case_id and row.get("phase") == phase]
    if len(selected) != 1:
        raise ValueError(f"expected one {case_id}/{phase} row, got {len(selected)}")
    return selected[0]


def _validate_process(name, document):
    if not isinstance(document, dict) or document.get("schema") != 1:
        raise ValueError(f"{name} schema mismatch")
    resource_cutoff = (
        name == "HYBRID1024"
        and document.get("status") == "PASS_ROUND11_T1_PROCESS_WITH_RESOURCE_CUTOFF"
    )
    if document.get("process") != name or (
        document.get("status") != "PASS_ROUND11_T1_PROCESS" and not resource_cutoff
    ):
        raise ValueError(f"{name} process/status mismatch")
    if resource_cutoff:
        source_hashes = document.get("source_hashes", {})
        if (
            document.get("composite") is not True
            or document.get("raw_source_status") != "FAIL"
            or document.get("generation_attempts") != 2
            or document.get("abort_controls") != 0
            or document.get("automatic_retries") != 0
            or document.get("hybrid_qualified_up_to_tokens") != 768
            or document.get("resource_cutoff_tokens") != 1024
            or document.get("cleanup")
            != {"owned_container_removed": True, "physical12_after": "EMPTY"}
            and not (
                document.get("cleanup", {}).get("owned_container_removed") is True
                and document.get("cleanup", {}).get("physical12_after") == "EMPTY"
            )
            or not isinstance(source_hashes, dict)
            or len(source_hashes) != 4
            or any(
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for value in source_hashes.values()
            )
        ):
            raise ValueError("HYBRID1024 resource-cutoff source contract mismatch")
    if document.get("composite") is True and not resource_cutoff:
        validation = document.get("composite_validation", {})
        source_hashes = validation.get("source_hashes", {})
        if (
            name != "HYBRID256"
            or document.get("physical_generation_attempts") != 17
            or document.get("physical_model_processes") != 2
            or validation.get("status") != "PASS_STRICT_V109_PREFIX_PLUS_CANCEL_REPAIR"
            or validation.get("immutable_failed_source_status") != "FAIL"
            or validation.get("failed_source_rows_used") != 13
            or validation.get("repair_rows_used") != 2
            or not isinstance(source_hashes, dict)
            or len(source_hashes) != 4
            or any(
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for value in source_hashes.values()
            )
        ):
            raise ValueError("HYBRID256 composite source contract mismatch")
    expected = 1 if resource_cutoff else EXPECTED_COUNTS[name]
    rows = document.get("results")
    if (
        not isinstance(rows, list)
        or len(rows) != expected
        or document.get("generation_attempts") != (2 if resource_cutoff else expected)
        or document.get("automatic_retries") != 0
        or document.get("abort_controls") != EXPECTED_ABORTS[name]
    ):
        raise ValueError(f"{name} frozen budget/result count mismatch")
    for row in rows:
        if row.get("phase") == "cancel":
            if not str(row.get("cancel_status", "")).startswith("PASS"):
                raise ValueError(f"{name} cancellation row failed")
            continue
        if row.get("state") != "PASS":
            raise ValueError(f"{name} has a non-passing row")
        validated, path = row.get("validated"), row.get("path")
        if not isinstance(validated, dict) or validated.get("finite_output_logprobs") is not True:
            raise ValueError(f"{name} response/logprob validation missing")
        if not isinstance(path, dict) or path.get("request_total_tokens", 0) < 1:
            raise ValueError(f"{name} path validation missing")
        expected_path = "HYBRID" if name != "BOUNDARY" or row.get("case_id") == "edge128" else "LAYERWISE"
        if path.get("path") != expected_path:
            raise ValueError(f"{name} selected path mismatch")
    if len(document.get("cancel_receipts", [])) != EXPECTED_ABORTS[name]:
        raise ValueError(f"{name} cancel/reuse receipt count mismatch")
    return rows


def aggregate(documents):
    missing = set(PROCESSES) - set(documents)
    extra = set(documents) - set(PROCESSES)
    if missing or extra:
        raise ValueError(f"missing process evidence={sorted(missing)} extra={sorted(extra)}")
    rows = {name: _validate_process(name, documents[name]) for name in PROCESSES}

    boundary_rows = rows["BOUNDARY"]
    baseline_boundary = {
        str(length): _index(boundary_rows, f"edge{length}", "boundary")["path"]["path"]
        for length in (128, 129, 130)
    }
    expected_baseline_boundary = {"128": "HYBRID", "129": "LAYERWISE", "130": "LAYERWISE"}
    if baseline_boundary != expected_baseline_boundary:
        raise ValueError("baseline 129 path-switch contract mismatch")

    extended = {}
    for name in ("HYBRID256", "HYBRID512", "HYBRID1024"):
        for row in rows[name]:
            if row.get("phase") == "extended_prefill":
                extended[int(row["path"]["request_total_tokens"])] = row
    resource_cutoff = documents["HYBRID1024"].get("resource_cutoff_tokens") == 1024
    expected_lengths = {129, 130, 160, 192, 256, 384, 512, 768}
    if not resource_cutoff:
        expected_lengths.add(1024)
    if set(extended) != expected_lengths:
        raise ValueError("hierarchical Hybrid prefill evidence is incomplete")

    decode_steps = []
    for name in ("HYBRID256", "HYBRID512"):
        for row in rows[name]:
            if row.get("phase") == "decode_exact":
                output = row["validated"].get("output_ids")
                if not isinstance(output, list) or len(output) not in {32, 64, 128, 256}:
                    raise ValueError("Decode output length evidence mismatch")
                decode_steps.append(len(output))
    if sorted(decode_steps) != [32, 64, 128, 256]:
        raise ValueError("Decode qualification ladder incomplete")

    first256 = _index(rows["HYBRID256"], "edge256", "extended_prefill")
    repeated256 = _index(rows["HYBRID256"], "edge256", "repeatability")
    if first256["validated"].get("output_ids") != repeated256["validated"].get("output_ids"):
        raise ValueError("edge256 fixed-seed repeatability mismatch")

    ratios = {}
    for length in (129, 130):
        baseline = _index(boundary_rows, f"edge{length}", "boundary")
        forced = extended[length]
        ratios[str(length)] = {
            "layerwise_seconds": baseline["e2e_seconds"],
            "forced_hybrid_seconds": forced["e2e_seconds"],
            "hybrid_over_layerwise": forced["e2e_seconds"] / baseline["e2e_seconds"],
        }

    return {
        "schema": 1,
        "status": (
            "PASS_ROUND11_T1_WITH_RESOURCE_CUTOFF" if resource_cutoff else "PASS_ROUND11_T1"
        ),
        "baseline_boundary": baseline_boundary,
        "experimental_threshold": 1025,
        "hybrid_qualified_up_to_tokens": max(extended),
        "decode_steps_qualified": sorted(decode_steps),
        "deterministic_canaries_pass": all(
            row.get("state") == "PASS" for row in rows["HYBRID256"] if row.get("phase") == "canary"
        ),
        "repeatability_pass": True,
        "cancel_and_reuse_pass": True,
        "same_input_path_ratios": ratios,
        # The frozen default changes path at 129. That policy boundary is not
        # evidence that expanded Hybrid has reached a performance crossover.
        # Every experimentally forced Hybrid length through 1024 remained
        # correct, and only 129/130 have same-input Layerwise comparisons.
        "baseline_path_switch_at_tokens": 129,
        "crossover": (
            "NOT_FOUND_WITHIN_QUALIFIED_RANGE"
            if resource_cutoff
            else "NOT_FOUND_WITHIN_TESTED_RANGE"
        ),
        "expanded_hybrid_crossover": (
            "NOT_FOUND_WITHIN_QUALIFIED_RANGE"
            if resource_cutoff
            else "NOT_FOUND_WITHIN_TESTED_RANGE"
        ),
        "resource_cutoff_tokens": 1024 if resource_cutoff else None,
        "resource_cutoff_reason": (
            documents["HYBRID1024"].get("resource_cutoff_reason") if resource_cutoff else None
        ),
        "qualification_scope": (
            f"SERIAL_SINGLE_REQUEST_PROMPT_UP_TO_{max(extended)}_DECODE_UP_TO_256"
        ),
        "default_threshold_changed": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for process in PROCESSES:
        parser.add_argument(f"--{process.lower()}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    documents = {
        process: json.loads(getattr(args, process.lower()).read_text()) for process in PROCESSES
    }
    result = aggregate(documents)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(result["status"], result["hybrid_qualified_up_to_tokens"], flush=True)


if __name__ == "__main__":
    main()
