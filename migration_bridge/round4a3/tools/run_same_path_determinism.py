#!/usr/bin/env python3
"""Verify repeated greedy outputs and 8/16/32-to-64 prefix consistency."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path


def post_json(url: str, payload: dict) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.load(response)


def generate(base_url: str, input_ids: list[int], max_new_tokens: int) -> list[int]:
    response = post_json(
        base_url.rstrip("/") + "/generate",
        {
            "input_ids": input_ids,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": max_new_tokens,
                "ignore_eos": True,
            },
        },
    )
    return [int(value) for value in response["output_ids"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:31000")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--skip-prefix-check",
        action="store_true",
        help="Do not issue the 8/16/32 token prefix requests.",
    )
    parser.add_argument(
        "--prompt-id",
        action="append",
        default=[],
        help="Run only this prompt ID; may be supplied more than once.",
    )
    parser.add_argument(
        "--append-input-id",
        type=int,
        action="append",
        default=[],
        help="Append a fixed teacher-forced token ID to every selected prompt.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    corpus = json.loads(args.corpus.read_text())
    selected_prompt_ids = set(args.prompt_id)
    prompts = [
        prompt
        for prompt in corpus["prompts"]
        if not selected_prompt_ids or prompt["id"] in selected_prompt_ids
    ]
    missing_prompt_ids = selected_prompt_ids - {prompt["id"] for prompt in prompts}
    if missing_prompt_ids:
        raise SystemExit(
            "unknown prompt ID(s): " + ", ".join(sorted(missing_prompt_ids))
        )
    if args.max_new_tokens <= 0:
        raise SystemExit("--max-new-tokens must be positive")
    if not args.skip_prefix_check and args.max_new_tokens < 32:
        raise SystemExit(
            "--max-new-tokens must be at least 32 unless --skip-prefix-check is used"
        )
    rows = []
    for prompt in prompts:
        started = time.perf_counter()
        input_ids = [int(value) for value in prompt["input_ids"]]
        input_ids.extend(args.append_input_id)
        runs = [
            generate(args.base_url, input_ids, args.max_new_tokens)
            for _ in range(args.repeats)
        ]
        short = (
            {}
            if args.skip_prefix_check
            else {
                str(length): generate(args.base_url, input_ids, length)
                for length in (8, 16, 32)
            }
        )
        repeat_exact = all(run == runs[0] for run in runs[1:])
        prefix_exact = args.skip_prefix_check or all(
            short[str(length)] == runs[0][:length] for length in (8, 16, 32)
        )
        rows.append(
            {
                "prompt_id": prompt["id"],
                "input_ids_sha256": hashlib.sha256(
                    json.dumps(input_ids, separators=(",", ":")).encode()
                ).hexdigest(),
                "repeat_exact": repeat_exact,
                "prefix_exact": prefix_exact,
                "hashes64": [hashlib.sha256(json.dumps(run).encode()).hexdigest() for run in runs],
                "hashes": [hashlib.sha256(json.dumps(run).encode()).hexdigest() for run in runs],
                "short_outputs": short,
                "reference64": runs[0],
                "reference": runs[0],
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        print(prompt["id"], repeat_exact, prefix_exact, flush=True)
    payload = {
        "schema_version": 1,
        "mode": args.mode,
        "corpus_sha256": corpus["sha256"],
        "prompt_ids": [prompt["id"] for prompt in prompts],
        "appended_input_ids": args.append_input_id,
        "repeats": args.repeats,
        "max_new_tokens": args.max_new_tokens,
        "prefix_check_enabled": not args.skip_prefix_check,
        "temperature": 0,
        "protocol_seed": 0,
        "all_repeat_exact": all(row["repeat_exact"] for row in rows),
        "all_prefix_exact": all(row["prefix_exact"] for row in rows),
        "rows": rows,
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    payload["sha256"] = hashlib.sha256(canonical).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    if not payload["all_repeat_exact"] or not payload["all_prefix_exact"]:
        raise SystemExit("NONDETERMINISM")


if __name__ == "__main__":
    main()
