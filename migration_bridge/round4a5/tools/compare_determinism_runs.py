#!/usr/bin/env python3
"""Compare two same-path determinism artifacts across a clean restart."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def canonical_sha256(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load(path: Path) -> dict:
    payload = json.loads(path.read_text())
    required = {
        "all_prefix_exact",
        "all_repeat_exact",
        "corpus_sha256",
        "rows",
        "sha256",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise RuntimeError(f"{path}: missing fields {missing}")
    return payload


def row_map(payload: dict, path: Path) -> dict[str, dict]:
    rows = {row["prompt_id"]: row for row in payload["rows"]}
    if len(rows) != len(payload["rows"]):
        raise RuntimeError(f"{path}: duplicate prompt_id")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    before = load(args.before)
    after = load(args.after)
    before_rows = row_map(before, args.before)
    after_rows = row_map(after, args.after)
    prompt_ids = list(before.get("prompt_ids", before_rows))
    after_prompt_ids = list(after.get("prompt_ids", after_rows))

    same_corpus = before["corpus_sha256"] == after["corpus_sha256"]
    same_prompt_ids = prompt_ids == after_prompt_ids
    same_row_ids = set(before_rows) == set(after_rows)
    prompt_results = {}
    for prompt_id in prompt_ids:
        left = before_rows.get(prompt_id)
        right = after_rows.get(prompt_id)
        left_input_hash = left.get("input_ids_sha256") if left is not None else None
        right_input_hash = right.get("input_ids_sha256") if right is not None else None
        input_ids_exact = bool(
            left is not None
            and right is not None
            and (
                left_input_hash == right_input_hash
                if left_input_hash is not None or right_input_hash is not None
                else True
            )
        )
        prompt_results[prompt_id] = {
            "present_in_both": left is not None and right is not None,
            "input_ids_exact": input_ids_exact,
            "token_ids_exact": left is not None
            and right is not None
            and left["reference64"] == right["reference64"],
        }

    before_internal = bool(before["all_repeat_exact"] and before["all_prefix_exact"])
    after_internal = bool(after["all_repeat_exact"])
    all_exact = bool(
        same_corpus
        and same_prompt_ids
        and same_row_ids
        and before_internal
        and after_internal
        and all(all(result.values()) for result in prompt_results.values())
    )
    payload = {
        "schema_version": 1,
        "before_sha256": before["sha256"],
        "after_sha256": after["sha256"],
        "same_corpus": same_corpus,
        "same_prompt_ids": same_prompt_ids,
        "same_row_ids": same_row_ids,
        "before_internal_exact": before_internal,
        "after_internal_exact": after_internal,
        "prompt_results": prompt_results,
        "all_exact": all_exact,
    }
    payload["sha256"] = canonical_sha256(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not all_exact:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
