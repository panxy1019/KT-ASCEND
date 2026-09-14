#!/usr/bin/env python3
"""Run one frozen downstream quality manifest against one execution mode."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
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
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.load(response)


def normalize_number(value: str) -> str:
    value = value.strip().replace(",", "")
    try:
        number = float(value)
    except ValueError:
        return value
    return str(int(number)) if number.is_integer() else format(number, ".12g")


def score(benchmark: str, text: str, reference: str) -> tuple[bool, bool, str | None]:
    if benchmark == "ceval":
        matches = re.findall(r"(?:^|[^A-Za-z])([ABCD])(?:[^A-Za-z]|$)", text.upper())
        parsed = matches[-1] if matches else None
    elif benchmark == "gsm8k":
        matches = re.findall(r"Final\s*answer\s*:\s*([-+]?\d[\d,]*(?:\.\d+)?)", text, flags=re.IGNORECASE)
        parsed = normalize_number(matches[-1]) if matches else None
        reference = normalize_number(reference)
    else:
        raise ValueError(f"unsupported benchmark: {benchmark}")
    return parsed is not None and parsed == reference, parsed is None, parsed


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
        max_new_tokens = int(manifest["protocol"][f"{benchmark}_max_new_tokens"])
        for sample in spec["samples"]:
            started = time.perf_counter()
            response = post_json(
                args.base_url.rstrip("/") + "/generate",
                {
                    "text": sample["prompt"],
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": max_new_tokens,
                    },
                },
            )
            text = response.get("text", "")
            correct, invalid, parsed = score(benchmark, text, sample["reference_answer"])
            rows.append(
                {
                    "benchmark": benchmark,
                    "sample_id": sample["id"],
                    "reference_answer": sample["reference_answer"],
                    "parsed_answer": parsed,
                    "correct": correct,
                    "invalid": invalid,
                    "output_ids": response["output_ids"],
                    "text": text,
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
        "all_output_tokens_valid": all(all(isinstance(token, int) for token in row["output_ids"]) for row in rows),
        "rows": rows,
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    payload["sha256"] = hashlib.sha256(canonical).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
