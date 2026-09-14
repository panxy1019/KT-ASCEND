#!/usr/bin/env python3
"""Locate first same-path divergence and output clusters in repeated runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def token_sha(tokens: list[int]) -> str:
    return hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.input.read_text())
    rows = []
    for row in source["rows"]:
        sequences = [item["output_ids"] for item in row["repetitions"]]
        reference = sequences[0]
        first = None
        alternatives = Counter()
        for sequence in sequences[1:]:
            for index, (expected, actual) in enumerate(zip(reference, sequence)):
                if actual != expected:
                    first = index if first is None else min(first, index)
                    alternatives[(index, expected, actual)] += 1
                    break
            else:
                if len(sequence) != len(reference):
                    index = min(len(sequence), len(reference))
                    first = index if first is None else min(first, index)
        hashes = Counter(token_sha(sequence) for sequence in sequences)
        rows.append(
            {
                "prompt_id": row["prompt_id"],
                "repeat_count": len(sequences),
                "unique_output_hashes": len(hashes),
                "hash_counts": dict(sorted(hashes.items())),
                "first_divergence_index": first,
                "first_divergence_alternatives": [
                    {"index": key[0], "reference": key[1], "actual": key[2], "count": count}
                    for key, count in sorted(alternatives.items())
                ],
            }
        )
    payload = {
        "schema_version": 1,
        "source_sha256": source["sha256"],
        "rows": rows,
        "deterministic": all(row["unique_output_hashes"] == 1 for row in rows),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["sha256"] = hashlib.sha256(canonical).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
