#!/usr/bin/env python3
"""Repeat 64-token greedy generations and compare with a frozen matrix."""

from __future__ import annotations

import argparse
import json
import math
import time
import urllib.request
from pathlib import Path


def post_json(url: str, payload: dict) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:31000")
    parser.add_argument("--cycles", default=3, type=int)
    args = parser.parse_args()
    if args.cycles <= 0:
        raise ValueError("--cycles must be positive")

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    expected = {
        case["prompt_name"]: case
        for case in baseline["cases"]
        if case["max_new_tokens"] == 64
    }
    if set(expected) != set(baseline["prompts"]):
        raise ValueError("baseline is missing one or more 64-token prompt cases")

    results = []
    for cycle in range(1, args.cycles + 1):
        for prompt_name, prompt in baseline["prompts"].items():
            started = time.perf_counter()
            response = post_json(
                f"{args.base_url}/generate",
                {
                    "text": prompt,
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 64,
                        "ignore_eos": True,
                    },
                    "return_logprob": True,
                    "logprob_start_len": 0,
                },
            )
            elapsed = time.perf_counter() - started
            ids = response.get("output_ids", [])
            logprobs = [
                item[0]
                for item in response.get("meta_info", {}).get(
                    "output_token_logprobs", []
                )
            ]
            ids_equal = ids == expected[prompt_name]["output_ids"]
            finite = len(logprobs) == 64 and all(
                value is not None and math.isfinite(value) for value in logprobs
            )
            result = {
                "cycle": cycle,
                "prompt_name": prompt_name,
                "elapsed_seconds": elapsed,
                "token_count": len(ids),
                "ids_equal_to_baseline": ids_equal,
                "all_logprobs_finite": finite,
            }
            results.append(result)
            print(result, flush=True)
            if not ids_equal or not finite:
                raise AssertionError(f"stability failure: {result}")

    output = {
        "cycles": args.cycles,
        "requests": len(results),
        "tokens_per_request": 64,
        "all_ids_equal_to_baseline": True,
        "all_logprobs_finite": True,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
