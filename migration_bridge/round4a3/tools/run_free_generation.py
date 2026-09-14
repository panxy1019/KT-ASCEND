#!/usr/bin/env python3
"""Run deterministic greedy generation for a frozen Round4A3 corpus."""

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:31000")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    corpus = json.loads(args.corpus.read_text())
    rows = []
    for prompt in corpus["prompts"]:
        repetitions = []
        for repeat in range(args.repeats):
            started = time.perf_counter()
            response = post_json(
                args.base_url.rstrip("/") + "/generate",
                {
                    "input_ids": prompt["input_ids"],
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": args.max_new_tokens,
                        "ignore_eos": True,
                    },
                    "return_logprob": True,
                    "logprob_start_len": 0,
                    "top_logprobs_num": 16,
                },
            )
            repetitions.append(
                {
                    "repeat": repeat,
                    "elapsed_seconds": time.perf_counter() - started,
                    "output_ids": response["output_ids"],
                    "output_logprobs": response["meta_info"]["output_token_logprobs"],
                    "output_top_logprobs": response["meta_info"]["output_top_logprobs"],
                    "text": response.get("text", ""),
                }
            )
        deterministic = all(row["output_ids"] == repetitions[0]["output_ids"] for row in repetitions[1:])
        rows.append({"prompt_id": prompt["id"], "deterministic": deterministic, "repetitions": repetitions})
        print(prompt["id"], deterministic, flush=True)
    payload = {
        "schema_version": 1,
        "mode": args.mode,
        "corpus_sha256": corpus["sha256"],
        "max_new_tokens": args.max_new_tokens,
        "repeats": args.repeats,
        "all_deterministic": all(row["deterministic"] for row in rows),
        "rows": rows,
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    payload["sha256"] = hashlib.sha256(canonical).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
