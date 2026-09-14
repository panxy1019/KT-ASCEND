#!/usr/bin/env python3
"""Freeze disjoint Round 4A prompt corpora with tokenizer IDs and SHA256."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


CORPORA = {
    "selection": [
        ("s_en_fact_01", "english_factual", "Name the largest ocean on Earth."),
        ("s_en_fact_02", "english_factual", "Which element has the chemical symbol Fe?"),
        ("s_en_fact_03", "english_factual", "In what year did the first Moon landing occur?"),
        ("s_zh_fact_01", "chinese_factual", "请说明长江发源于哪个地区。"),
        ("s_zh_fact_02", "chinese_factual", "中国古代四大发明包括哪些？"),
        ("s_zh_fact_03", "chinese_factual", "水在标准大气压下的沸点是多少摄氏度？"),
        ("s_math_01", "math_numeric", "Compute 37 * 24 and show the result."),
        ("s_math_02", "math_numeric", "A train travels 180 km in 2.5 hours. What is its average speed?"),
        ("s_json_01", "structured_json", "Return JSON with keys city and country for Tokyo."),
        ("s_json_02", "structured_json", "Convert Alice, 31, engineer into a compact JSON object."),
        ("s_code_01", "code_reasoning", "Write a Python function that returns the median of a list."),
        ("s_code_02", "code_reasoning", "Explain why binary search requires sorted input."),
    ],
    "validation": [
        ("v_en_01", "english", "Why do leaves usually appear green?"),
        ("v_en_02", "english", "Who wrote Pride and Prejudice?"),
        ("v_en_03", "english", "Summarize the purpose of a database index in one sentence."),
        ("v_zh_01", "chinese", "请简要解释什么是光合作用。"),
        ("v_zh_02", "chinese", "北京位于中国的哪个方向？"),
        ("v_zh_03", "chinese", "请列出春夏秋冬四个季节。"),
        ("v_struct_01", "structured_math_code", "Sequence: 3, 6, 12, 24, next:"),
        ("v_struct_02", "structured_math_code", "Return a JSON array containing the first five prime numbers."),
        ("v_struct_03", "structured_math_code", "Write pseudocode to test whether an integer is even."),
    ],
    "stability": [
        ("t_01", "stability", "Describe how rain forms in the atmosphere."),
        ("t_02", "stability", "请用两句话介绍故宫。"),
        ("t_03", "stability", "Calculate the sum of the integers from 1 through 20."),
        ("t_04", "stability", "Create JSON for three colors and their hexadecimal codes."),
        ("t_05", "stability", "Write a short Python loop that prints squares from 1 to 5."),
        ("t_06", "stability", "Explain the difference between RAM and persistent storage."),
    ],
}


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def freeze(model_dir: Path, output_dir: Path) -> dict[str, dict]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir.resolve()), local_files_only=True, trust_remote_code=False
    )
    seen_prompts: set[str] = set()
    frozen: dict[str, dict] = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    for corpus_name, rows in CORPORA.items():
        entries = []
        for prompt_id, category, text in rows:
            if text in seen_prompts:
                raise AssertionError(f"prompt reused across corpora: {text}")
            seen_prompts.add(text)
            entries.append(
                {
                    "id": prompt_id,
                    "category": category,
                    "text": text,
                    "input_ids": tokenizer.encode(text, add_special_tokens=True),
                }
            )
        payload = {
            "corpus": corpus_name,
            "count": len(entries),
            "model_dir": str(model_dir.resolve()),
            "tokenizer_class": type(tokenizer).__name__,
            "prompts": entries,
        }
        payload["sha256"] = sha256(payload)
        (output_dir / f"corpus_{corpus_name}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        frozen[corpus_name] = payload
    return frozen


def main() -> None:
    args = parse_args()
    frozen = freeze(args.model_dir, args.output_dir)
    print(
        json.dumps(
            {name: {"count": data["count"], "sha256": data["sha256"]} for name, data in frozen.items()},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
