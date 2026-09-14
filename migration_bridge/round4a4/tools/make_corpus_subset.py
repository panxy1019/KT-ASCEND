#!/usr/bin/env python3
"""Freeze an ordered prompt subset from an already-frozen corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def canonical_sha(payload: dict) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--prompt-ids", required=True)
    parser.add_argument("--purpose", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.source.read_text())
    requested = [value for value in args.prompt_ids.split(",") if value]
    by_id = {prompt["id"]: prompt for prompt in source["prompts"]}
    missing = [prompt_id for prompt_id in requested if prompt_id not in by_id]
    if missing:
        raise RuntimeError(f"missing prompt IDs: {missing}")
    payload = {
        "schema_version": 1,
        "purpose": args.purpose,
        "source_corpus_sha256": source["sha256"],
        "prompts": [by_id[prompt_id] for prompt_id in requested],
    }
    payload["sha256"] = canonical_sha(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(payload["sha256"])


if __name__ == "__main__":
    main()
