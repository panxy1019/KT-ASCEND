"""Compose v109's passing prefix with a separately verified cancel/reuse repair run.

The failed v109 document remains immutable.  The output is an explicit composite
qualification document whose physical request count is larger than its logical
15-row schedule coverage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


FAILED_ERROR = "RuntimeError('cancel fence/cleanup/retire evidence incomplete')"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _validate_supervisor(document, *, status, requests, driver_exit):
    if document.get("status") != status or document.get("driver_exit") != driver_exit:
        raise ValueError("supervisor status/driver exit mismatch")
    if document.get("requests_recorded") != requests or document.get("generation_driver_starts") != 1:
        raise ValueError("supervisor request/start count mismatch")
    if document.get("owned_container_removed") is not True:
        raise ValueError("supervisor container cleanup missing")
    state = document.get("owned_exit_state", {})
    if state.get("OOMKilled") is not False or state.get("Running") is not False:
        raise ValueError("supervisor owned exit state invalid")
    if document.get("guard_exit_ack", {}).get("action") not in {
        "OWNED_EXITED", "OWNED_EXITED_DURING_SAMPLE"
    }:
        raise ValueError("supervisor guard ACK missing")


def _validate_common_row(row):
    if row.get("state") != "PASS":
        raise ValueError("passing source row expected")
    validated = row.get("validated", {})
    if validated.get("finite_output_logprobs") is not True:
        raise ValueError("finite-logprob source evidence missing")
    if row.get("path", {}).get("path") != "HYBRID":
        raise ValueError("source row did not use Hybrid")


def compose(failed, failed_supervisor, repair, repair_supervisor, *, source_hashes=None):
    if (
        failed.get("schema") != 1
        or failed.get("process") != "HYBRID256"
        or failed.get("status") != "FAIL"
        or failed.get("error") != FAILED_ERROR
        or failed.get("generation_attempts") != 15
        or failed.get("abort_controls") != 1
        or failed.get("automatic_retries") != 0
    ):
        raise ValueError("v109 frozen failure identity mismatch")
    rows = failed.get("results", [])
    if len(rows) != 15:
        raise ValueError("v109 result row count mismatch")
    for row in rows[:13]:
        _validate_common_row(row)
    if rows[13].get("phase") != "cancel" or rows[13].get("cancel_status") != "PENDING_REUSE_VALIDATION":
        raise ValueError("v109 cancellation failure boundary mismatch")
    if rows[14].get("phase") != "reuse":
        raise ValueError("v109 reuse row missing")
    _validate_common_row(rows[14])
    _validate_supervisor(failed_supervisor, status="FAIL", requests=15, driver_exit=1)

    if (
        repair.get("schema") != 1
        or repair.get("process") != "HYBRID_CANCEL_REPAIR"
        or repair.get("status") != "PASS_ROUND11_T1_PROCESS"
        or repair.get("generation_attempts") != 2
        or repair.get("abort_controls") != 1
        or repair.get("automatic_retries") != 0
    ):
        raise ValueError("cancel/reuse repair identity mismatch")
    repair_rows = repair.get("results", [])
    if len(repair_rows) != 2:
        raise ValueError("cancel/reuse repair row count mismatch")
    cancel, reuse = repair_rows
    if cancel.get("phase") != "cancel" or not str(cancel.get("cancel_status", "")).startswith("PASS"):
        raise ValueError("cancel/reuse repair cancellation did not pass")
    if reuse.get("phase") != "reuse":
        raise ValueError("cancel/reuse repair reuse row missing")
    _validate_common_row(reuse)
    receipts = repair.get("cancel_receipts", [])
    if len(receipts) != 1:
        raise ValueError("cancel/reuse repair receipt count mismatch")
    _validate_supervisor(repair_supervisor, status="PASS", requests=2, driver_exit=0)

    combined_rows = [*rows[:13], cancel, reuse]
    repeats = [
        item["validated"]["output_ids"]
        for item in combined_rows
        if item.get("case_id") == "edge256"
        and item.get("phase") in {"extended_prefill", "repeatability"}
    ]
    if len(repeats) != 2 or repeats[0] != repeats[1]:
        raise ValueError("v109 edge256 repeatability mismatch")

    result = {
        key: failed[key]
        for key in (
            "schema", "baseline", "experimental_profile", "fixture_sha256",
            "threshold_tokens", "chunk_tokens", "selection_basis", "profiler",
            "schedule", "started_wall",
        )
    }
    result.update(
        status="PASS_ROUND11_T1_PROCESS",
        process="HYBRID256",
        generation_budget=15,
        generation_attempts=15,
        abort_controls=1,
        automatic_retries=0,
        results=combined_rows,
        cancel_receipts=receipts,
        edge256_repeat_exact=True,
        ended_wall=repair.get("ended_wall"),
        composite=True,
        physical_generation_attempts=17,
        physical_model_processes=2,
        composite_validation={
            "status": "PASS_STRICT_V109_PREFIX_PLUS_CANCEL_REPAIR",
            "immutable_failed_source_status": "FAIL",
            "failed_source_error": FAILED_ERROR,
            "failed_source_rows_used": 13,
            "repair_rows_used": 2,
            "source_hashes": dict(source_hashes or {}),
        },
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failed", type=Path, required=True)
    parser.add_argument("--failed-supervisor", type=Path, required=True)
    parser.add_argument("--repair", type=Path, required=True)
    parser.add_argument("--repair-supervisor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    paths = {
        "v109_failed_result_sha256": args.failed,
        "v109_failed_supervisor_sha256": args.failed_supervisor,
        "repair_result_sha256": args.repair,
        "repair_supervisor_sha256": args.repair_supervisor,
    }
    result = compose(
        json.loads(args.failed.read_text()),
        json.loads(args.failed_supervisor.read_text()),
        json.loads(args.repair.read_text()),
        json.loads(args.repair_supervisor.read_text()),
        source_hashes={name: _sha(path) for name, path in paths.items()},
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(result["composite_validation"]["status"], flush=True)


if __name__ == "__main__":
    main()
