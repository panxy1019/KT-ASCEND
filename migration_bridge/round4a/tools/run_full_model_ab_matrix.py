#!/usr/bin/env python3
"""Run a frozen-corpus greedy full-model matrix and emit auditable JSON."""

from __future__ import annotations

import argparse
import json
import math
import time
import urllib.request
from pathlib import Path


DEFAULT_TOKEN_COUNTS = (1, 8, 16, 32, 64)


def post_json(url: str, payload: dict | None = None):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload or {}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        body = response.read().decode("utf-8")
    if not body:
        return None
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return body


def parse_counts(value: str) -> tuple[int, ...]:
    counts = tuple(int(item) for item in value.split(",") if item)
    if not counts or any(count <= 0 for count in counts):
        raise argparse.ArgumentTypeError("token counts must be positive")
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:31000")
    parser.add_argument("--mode", required=True)
    parser.add_argument("--profile", required=True, choices=("all_npu", "p0", "p1", "p2", "p3"))
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--token-counts", default=DEFAULT_TOKEN_COUNTS, type=parse_counts)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
    cases = []
    for prompt in corpus["prompts"]:
        for max_new_tokens in args.token_counts:
            started = time.perf_counter()
            response = post_json(
                f"{args.base_url}/generate",
                {
                    "text": prompt["text"],
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": max_new_tokens,
                        "ignore_eos": True,
                    },
                    "return_logprob": True,
                    "logprob_start_len": 0,
                },
            )
            if not isinstance(response, dict):
                raise RuntimeError(f"unexpected response for {prompt['id']}: {response!r}")
            output_ids = response.get("output_ids", [])
            raw_logprobs = response.get("meta_info", {}).get("output_token_logprobs", [])
            logprobs = [item[0] for item in raw_logprobs]
            if len(output_ids) != max_new_tokens or len(logprobs) != max_new_tokens:
                raise AssertionError(f"{prompt['id']}/{max_new_tokens}: incomplete output")
            if not all(value is not None and math.isfinite(value) for value in logprobs):
                raise AssertionError(f"{prompt['id']}/{max_new_tokens}: non-finite logprob")
            case = {
                "prompt_id": prompt["id"],
                "category": prompt["category"],
                "prompt": prompt["text"],
                "max_new_tokens": max_new_tokens,
                "elapsed_seconds": time.perf_counter() - started,
                "output_ids": output_ids,
                "output_logprobs": logprobs,
                "text": response.get("text"),
            }
            cases.append(case)
            print(
                f"{prompt['id']:18s} tokens={max_new_tokens:2d} "
                f"elapsed={case['elapsed_seconds']:.3f}s first_id={output_ids[0]}",
                flush=True,
            )
    result = {
        "mode": args.mode,
        "profile": args.profile,
        "corpus": corpus["corpus"],
        "corpus_sha256": corpus["sha256"],
        "temperature": 0,
        "ignore_eos": True,
        "token_counts": list(args.token_counts),
        "request_count": len(cases),
        "all_outputs_finite": True,
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
