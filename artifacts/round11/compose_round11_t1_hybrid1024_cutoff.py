"""Freeze the v111 guard stop without rewriting its raw FAIL result."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _is_sha256(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_passed_768(row):
    validated = row.get("validated", {})
    path = row.get("path", {})
    dispatch = path.get("actual_dispatch_tokens")
    if (
        row.get("case_id") != "edge768"
        or row.get("phase") != "extended_prefill"
        or row.get("state") != "PASS"
        or validated.get("output_ids") != [3897, 248046]
        or validated.get("finite_output_logprobs") is not True
        or validated.get("normal_eos") is not True
        or path.get("path") != "HYBRID"
        or path.get("request_total_tokens") != 768
        or dispatch != [32] * 24
        or path.get("hybrid_provider_event_count", 0) <= 0
        or path.get("layerwise_event_count") != 0
    ):
        raise ValueError("v111 passed 768-token evidence mismatch")


def _parse_events(raw: str):
    rows = []
    for line in raw.splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def compose(result_path: Path, supervisor_path: Path, monitor_path: Path, cleanup_path: Path):
    result = json.loads(result_path.read_text())
    supervisor = json.loads(supervisor_path.read_text())
    monitor = _parse_events(monitor_path.read_text())
    cleanup_raw = cleanup_path.read_text()

    rows = result.get("results")
    if (
        result.get("schema") != 1
        or result.get("status") != "FAIL"
        or result.get("process") != "HYBRID1024"
        or result.get("generation_budget") != 2
        or result.get("generation_attempts") != 2
        or result.get("abort_controls") != 0
        or result.get("automatic_retries") != 0
        or not isinstance(rows, list)
        or len(rows) != 1
        or "RemoteDisconnected" not in str(result.get("error"))
    ):
        raise ValueError("v111 immutable raw failure contract mismatch")
    _validate_passed_768(rows[0])

    state = supervisor.get("owned_exit_state", {})
    ack = supervisor.get("guard_exit_ack", {})
    container_id = supervisor.get("container_id")
    if (
        supervisor.get("status") != "FAIL"
        or supervisor.get("process") != "HYBRID1024"
        or supervisor.get("version") != 111
        or supervisor.get("generation_driver_starts") != 1
        or not _is_sha256(container_id)
        or "server exited before Round11 T1 completion" not in str(supervisor.get("error"))
        or state.get("Status") != "exited"
        or state.get("Running") is not False
        or state.get("OOMKilled") is not False
        or state.get("ExitCode") != 137
        or ack.get("action") != "OWNED_EXITED"
        or ack.get("container_id") != container_id
        or ack.get("after_guard_stop") is not True
        or ack.get("state", {}).get("ExitCode") != 137
        or supervisor.get("owned_container_removed") is not True
    ):
        raise ValueError("v111 supervisor/cleanup contract mismatch")

    stop_rows = [row for row in monitor if row.get("action") == "STOP_OWNED"]
    exit_rows = [row for row in monitor if row.get("action") == "OWNED_EXITED"]
    samples = [row for row in monitor if "hbm_used_mib" in row]
    reason = stop_rows[0].get("reason") if len(stop_rows) == 1 else None
    if (
        reason != "NUMA4 free and conservative reclaim headroom below reserve"
        or len(exit_rows) != 1
        or exit_rows[0].get("container_id") != container_id
        or exit_rows[0].get("after_guard_stop") is not True
        or not samples
        or max(row["hbm_used_mib"] for row in samples) > 56832
        or max(row["owned_process_mib"] for row in samples) > 37376
    ):
        raise ValueError("v111 resource cutoff evidence mismatch")
    if "No process in device" not in cleanup_raw:
        raise ValueError("v111 post-cleanup physical12 evidence mismatch")

    return {
        "schema": 1,
        "status": "PASS_ROUND11_T1_PROCESS_WITH_RESOURCE_CUTOFF",
        "process": "HYBRID1024",
        "composite": True,
        "raw_source_status": "FAIL",
        "raw_generation_attempts": 2,
        "generation_attempts": 2,
        "abort_controls": 0,
        "automatic_retries": 0,
        "results": rows,
        "cancel_receipts": [],
        "hybrid_qualified_up_to_tokens": 768,
        "resource_cutoff_tokens": 1024,
        "resource_cutoff_reason": reason,
        "cleanup": {
            "guard_exit_ack": True,
            "oom_killed": False,
            "owned_container_removed": True,
            "physical12_after": "EMPTY",
        },
        "source_hashes": {
            "raw_result": sha256(result_path),
            "raw_supervisor": sha256(supervisor_path),
            "raw_monitor": sha256(monitor_path),
            "post_cleanup_proc_mem": sha256(cleanup_path),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--supervisor", type=Path, required=True)
    parser.add_argument("--monitor", type=Path, required=True)
    parser.add_argument("--cleanup", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    document = compose(args.result, args.supervisor, args.monitor, args.cleanup)
    args.output.write_text(json.dumps(document, indent=2) + "\n")
    print(document["status"], document["hybrid_qualified_up_to_tokens"], flush=True)


if __name__ == "__main__":
    main()
