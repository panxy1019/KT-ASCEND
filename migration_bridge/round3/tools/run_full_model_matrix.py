#!/usr/bin/env python3
"""Run the deterministic Round 3 full-model prompt/decode matrix."""

from __future__ import annotations

import argparse
import json
import math
import time
import urllib.request
from pathlib import Path


PROMPTS = {
    "english": "The capital of France is",
    "chinese": "中国的首都是",
    "structured_numeric": "Sequence: 2, 4, 6, 8, next:",
}
TOKEN_COUNTS = (1, 8, 16, 32, 64)


def post_json(url: str, payload: dict | None = None):
    data = json.dumps(payload or {}).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        body = response.read().decode("utf-8")
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return body


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:31000")
    parser.add_argument("--mode", required=True, choices=("all_npu", "hybrid"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--record-experts", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.record_experts:
        post_json(f"{args.base_url}/start_expert_distribution_record")

    cases = []
    try:
        for prompt_name, prompt in PROMPTS.items():
            for max_new_tokens in TOKEN_COUNTS:
                payload = {
                    "text": prompt,
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": max_new_tokens,
                        "ignore_eos": True,
                    },
                    "return_logprob": True,
                    "logprob_start_len": 0,
                }
                started = time.perf_counter()
                response = post_json(f"{args.base_url}/generate", payload)
                elapsed = time.perf_counter() - started
                if not isinstance(response, dict):
                    raise RuntimeError(f"unexpected response: {response!r}")
                output_ids = response.get("output_ids", [])
                output_logprobs = response.get("meta_info", {}).get(
                    "output_token_logprobs", []
                )
                if len(output_ids) != max_new_tokens:
                    raise AssertionError(
                        f"{prompt_name}/{max_new_tokens}: got {len(output_ids)} tokens"
                    )
                scalar_logprobs = [item[0] for item in output_logprobs]
                if len(scalar_logprobs) != max_new_tokens or not all(
                    value is not None and math.isfinite(value)
                    for value in scalar_logprobs
                ):
                    raise AssertionError(
                        f"{prompt_name}/{max_new_tokens}: non-finite/missing logprob"
                    )
                case = {
                    "prompt_name": prompt_name,
                    "prompt": prompt,
                    "max_new_tokens": max_new_tokens,
                    "elapsed_seconds": elapsed,
                    "output_ids": output_ids,
                    "output_logprobs": scalar_logprobs,
                    "text": response.get("text"),
                    "finish_reason": response.get("meta_info", {}).get(
                        "finish_reason"
                    ),
                    "prompt_tokens": response.get("meta_info", {}).get(
                        "prompt_tokens"
                    ),
                }
                cases.append(case)
                print(
                    f"{prompt_name:18s} tokens={max_new_tokens:2d} "
                    f"elapsed={elapsed:.3f}s first_id={output_ids[0]}"
                )
    finally:
        if args.record_experts:
            post_json(f"{args.base_url}/stop_expert_distribution_record")
            post_json(f"{args.base_url}/dump_expert_distribution_record")

    result = {
        "mode": args.mode,
        "base_url": args.base_url,
        "temperature": 0,
        "ignore_eos": True,
        "token_counts": list(TOKEN_COUNTS),
        "prompts": PROMPTS,
        "all_outputs_finite": True,
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
