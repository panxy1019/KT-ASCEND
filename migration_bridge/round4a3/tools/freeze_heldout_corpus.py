#!/usr/bin/env python3
"""Freeze Round4A3 Q and independent H corpora with tokenizer evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer


H_PROMPTS = (
    ("h_en_fact_01", "english_factual", "What is the capital city of Canada?"),
    ("h_en_fact_02", "english_factual", "Which scientist formulated the three laws of motion?"),
    ("h_en_fact_03", "english_factual", "Which organ pumps blood through the human body?"),
    ("h_zh_fact_01", "chinese_factual", "珠穆朗玛峰位于哪两个国家的边界？"),
    ("h_zh_fact_02", "chinese_factual", "《红楼梦》的作者通常被认为是谁？"),
    ("h_zh_fact_03", "chinese_factual", "地球绕太阳公转一周大约需要多长时间？"),
    ("h_math_01", "math_numeric", "Solve 7x - 5 = 44 and give the value of x."),
    ("h_math_02", "math_numeric", "A rectangle is 13 cm long and 8 cm wide. What is its area?"),
    ("h_json_01", "structured_json", "Return only a JSON object with keys name, age, and active for Mei, 28, true."),
    ("h_json_02", "structured_json", "Return only a JSON array containing the squares of 2, 3, and 4."),
    ("h_code_01", "code_reasoning", "Write a Python function that checks whether a string is a palindrome."),
    ("h_code_02", "code_reasoning", "Explain an algorithm to detect a cycle in a singly linked list."),
)


def canonical_sha(payload: dict) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def finalize(name: str, prompts: list[dict], model_dir: Path, tokenizer) -> dict:
    rows = []
    for prompt in prompts:
        row = dict(prompt)
        token_ids = tokenizer.encode(row["text"], add_special_tokens=False)
        if "input_ids" in row and row["input_ids"] != token_ids:
            raise RuntimeError(f"tokenizer drift for {row['id']}: {row['input_ids']} != {token_ids}")
        row["input_ids"] = token_ids
        rows.append(row)
    payload = {
        "schema_version": 1,
        "corpus": name,
        "purpose": "envelope_derivation" if name == "Q" else "independent_heldout",
        "model_dir": str(model_dir),
        "tokenizer_class": tokenizer.__class__.__name__,
        "tokenizer_revision": "DeepSeek-V2-Lite-604d5664 frozen local artifact",
        "protocol": {"teacher_forced_positions": 64, "top_k": 16, "temperature": 0, "seed": 0},
        "count": len(rows),
        "prompts": rows,
    }
    payload["sha256"] = canonical_sha(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--q-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    q_source = json.loads(args.q_source.read_text())
    q_prompts = [
        {"id": row["id"], "category": row["category"], "text": row["text"], "input_ids": row["input_ids"]}
        for row in q_source["prompts"]
    ]
    h_prompts = [{"id": key, "category": category, "text": text} for key, category, text in H_PROMPTS]
    q = finalize("Q", q_prompts, args.model_dir, tokenizer)
    h = finalize("H", h_prompts, args.model_dir, tokenizer)
    q_texts = {row["text"] for row in q["prompts"]}
    h_texts = {row["text"] for row in h["prompts"]}
    if q_texts & h_texts:
        raise RuntimeError("Q/H prompt overlap")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "q.json").write_text(json.dumps(q, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    (args.output_dir / "h.json").write_text(json.dumps(h, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"Q": q["sha256"], "H": h["sha256"]}, indent=2))


if __name__ == "__main__":
    main()
