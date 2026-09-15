#!/usr/bin/env python3
"""Find a bounded diagnostic request with no D1 CPU route in its full trace."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path


def post(base_url: str, input_ids: list[int]) -> dict:
    payload = {
        "input_ids": input_ids,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 1,
            "ignore_eos": True,
        },
        "return_logprob": True,
        "logprob_start_len": 0,
        "top_logprobs_num": 16,
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.load(response)


def read_records(path: Path, offset: int) -> list[dict]:
    with path.open("rb") as stream:
        stream.seek(offset)
        data = stream.read()
    if not data or not data.endswith(b"\n"):
        raise RuntimeError("missing or incomplete Deferred trace records")
    return [json.loads(line) for line in data.splitlines()]


def summarize(records: list[dict]) -> dict:
    submits = [row for row in records if row.get("event") == "submit"]
    pending = [row for row in records if row.get("event") == "pending_state"]
    if not submits or not pending:
        raise RuntimeError("request lacks Deferred submit/pending trace")
    return {
        "submit_count": len(submits),
        "immediate_cpu_routes": sum(int(row["immediate_cpu_routes"]) for row in submits),
        "deferred_cpu_routes": sum(int(row["deferred_cpu_routes"]) for row in submits),
        "pending_final": pending[-1]["pending_layers"],
        "trace_sequence_start": int(records[0].get("trace_sequence", 0)),
        "trace_sequence_end": int(records[-1].get("trace_sequence", 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:31000")
    parser.add_argument("--trace-file", type=Path, required=True)
    parser.add_argument("--start-token-id", type=int, default=1000)
    parser.add_argument("--budget", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.budget <= 0 or args.budget > 256:
        raise SystemExit("budget must be in 1..256")

    attempts = []
    selected = None
    for token_id in range(args.start_token_id, args.start_token_id + args.budget):
        offset = args.trace_file.stat().st_size if args.trace_file.exists() else 0
        response = post(args.base_url, [token_id])
        summary = summarize(read_records(args.trace_file, offset))
        row = {
            "input_ids": [token_id],
            "response_output_ids": [int(value) for value in response["output_ids"]],
            **summary,
        }
        row["whole_request_cpu_not_hit"] = (
            row["immediate_cpu_routes"] == 0
            and row["deferred_cpu_routes"] == 0
            and row["pending_final"] == []
        )
        attempts.append(row)
        if row["whole_request_cpu_not_hit"]:
            selected = row
            break

    result = {
        "schema_version": 1,
        "search_rule": "ascending_single_token_ids_first_match",
        "start_token_id": args.start_token_id,
        "predeclared_budget": args.budget,
        "attempt_count": len(attempts),
        "attempts": attempts,
        "selected": selected,
        "status": "PASS_CANDIDATE_FOUND" if selected else "NOT_OBSERVED",
    }
    canonical = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    result["sha256"] = hashlib.sha256(canonical).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
