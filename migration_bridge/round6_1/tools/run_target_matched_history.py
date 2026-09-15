#!/usr/bin/env python3
"""Run one frozen matched-history request for a Round 6.1 target position."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path


def canonical_sha(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def output_ids(row: dict) -> list[int]:
    if "repetitions" in row:
        return [int(value) for value in row["repetitions"][0]["output_ids"]]
    return [int(value) for value in row["reference64"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:31000")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--baseline-history", type=Path, required=True)
    parser.add_argument("--prompt-id", required=True)
    parser.add_argument("--token-index", type=int, required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    corpus = json.loads(args.corpus.read_text())
    history = json.loads(args.baseline_history.read_text())
    prompt = next(row for row in corpus["prompts"] if row["id"] == args.prompt_id)
    history_row = next(row for row in history["rows"] if row["prompt_id"] == args.prompt_id)
    generated = output_ids(history_row)
    if args.token_index >= len(generated):
        raise SystemExit("target token index is outside frozen baseline history")

    teacher_forced = [int(value) for value in generated[: args.token_index]]
    input_ids = [int(value) for value in prompt["input_ids"]] + teacher_forced
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
        args.base_url.rstrip("/") + "/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        result = json.load(response)

    record = {
        "schema_version": 1,
        "mode": args.mode,
        "corpus_sha256": corpus["sha256"],
        "baseline_history_sha256": history["sha256"],
        "prompt_id": args.prompt_id,
        "token_index": args.token_index,
        "expected_baseline_token": int(generated[args.token_index]),
        "teacher_forced_history_sha256": canonical_sha(teacher_forced),
        "request_input_ids_sha256": canonical_sha(input_ids),
        "request_input_length": len(input_ids),
        "response_output_ids": [int(value) for value in result["output_ids"]],
        "response": result,
    }
    canonical = json.dumps(
        record, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    record["sha256"] = hashlib.sha256(canonical).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
