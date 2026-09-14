#!/usr/bin/env python3
"""Validate repeated greedy identity and 8/16/32 prefixes of a 64-token run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat64", type=Path, required=True)
    parser.add_argument("--run8", type=Path, required=True)
    parser.add_argument("--run16", type=Path, required=True)
    parser.add_argument("--run32", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repeat64 = json.loads(args.repeat64.read_text())
    short = {
        8: json.loads(args.run8.read_text()),
        16: json.loads(args.run16.read_text()),
        32: json.loads(args.run32.read_text()),
    }
    base = {row["prompt_id"]: row for row in repeat64["rows"]}
    rows = []
    for prompt_id, row in base.items():
        reference = row["repetitions"][0]["output_ids"]
        repeated_exact = all(
            repetition["output_ids"] == reference for repetition in row["repetitions"]
        )
        prefixes = {}
        for length, run in short.items():
            candidate = next(item for item in run["rows"] if item["prompt_id"] == prompt_id)
            prefixes[str(length)] = candidate["repetitions"][0]["output_ids"] == reference[:length]
        rows.append(
            {
                "prompt_id": prompt_id,
                "repeat_count": len(row["repetitions"]),
                "repeated_exact": repeated_exact,
                "prefixes": prefixes,
            }
        )
    passed = all(row["repeated_exact"] and all(row["prefixes"].values()) for row in rows)
    payload = {
        "schema_version": 1,
        "rows": rows,
        "all_repeated_exact": all(row["repeated_exact"] for row in rows),
        "all_prefixes_exact": all(all(row["prefixes"].values()) for row in rows),
        "status": "A3_VERIFIED_READY" if passed else "SAME_PATH_NONDETERMINISM",
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["sha256"] = hashlib.sha256(canonical).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
