#!/usr/bin/env python3
"""Analyze one frozen Round17 JSONL profile without loading model code."""

import argparse
import json
from pathlib import Path

from round17.profile_analysis import analyze_profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spans", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = [json.loads(line) for line in args.spans.read_text().splitlines() if line]
    result = analyze_profile(rows)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(result["status"], result["steady_decode_batches"])


if __name__ == "__main__":
    main()
