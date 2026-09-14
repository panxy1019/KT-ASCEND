#!/usr/bin/env python3
"""Run one frozen downstream quality manifest against one serving mode."""

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
    with urllib.request.urlopen(request, timeout=900) as response:
        return json.load(response)


def score_four_choice(
    base_url: str, prompt: str
) -> tuple[str, dict[str, float], dict[str, list[int]]]:
    scores = {}
    token_ids = {}
    prefix = prompt + "\n答案："
    for choice in "ABCD":
        response = post_json(
            base_url.rstrip("/") + "/generate",
            {
                "text": prefix + choice,
                "sampling_params": {"temperature": 0, "max_new_tokens": 1},
                "return_logprob": True,
                "logprob_start_len": 0,
                "top_logprobs_num": 0,
            },
        )
        input_logprobs = response["meta_info"]["input_token_logprobs"]
        if not input_logprobs or input_logprobs[-1][0] is None:
            raise RuntimeError(f"missing conditional logprob for choice {choice}")
        scores[choice] = float(input_logprobs[-1][0])
        token_ids[choice] = [int(input_logprobs[-1][1])]
    selected = max(scores, key=scores.get)
    return selected, scores, token_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:31000")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    rows = []
    for benchmark, spec in manifest["benchmarks"].items():
        for sample in spec["samples"]:
            started = time.perf_counter()
            parsed, choice_scores, choice_token_ids = score_four_choice(
                args.base_url, sample["prompt"]
            )
            text = parsed
            correct = parsed == sample["reference_answer"]
            invalid = False
            output_ids = [choice_token_ids[parsed][-1]]
            rows.append(
                {
                    "benchmark": benchmark,
                    "sample_id": sample["id"],
                    "reference_answer": sample["reference_answer"],
                    "parsed_answer": parsed,
                    "correct": correct,
                    "invalid": invalid,
                    "output_ids": output_ids,
                    "text": text,
                    "choice_logprobs": choice_scores,
                    "choice_token_ids": choice_token_ids,
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            print(benchmark, sample["id"], correct, invalid, flush=True)
    summaries = {}
    for benchmark in manifest["benchmarks"]:
        selected = [row for row in rows if row["benchmark"] == benchmark]
        summaries[benchmark] = {
            "count": len(selected),
            "correct": sum(row["correct"] for row in selected),
            "score": sum(row["correct"] for row in selected) / len(selected),
            "invalid": sum(row["invalid"] for row in selected),
        }
    payload = {
        "schema_version": 1,
        "mode": args.mode,
        "quality_manifest_sha256": manifest["sha256"],
        "temperature": 0,
        "protocol_seed": 0,
        "summaries": summaries,
        "all_output_tokens_valid": all(
            all(isinstance(token, int) for token in row["output_ids"]) for row in rows
        ),
        "rows": rows,
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    payload["sha256"] = hashlib.sha256(canonical).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
