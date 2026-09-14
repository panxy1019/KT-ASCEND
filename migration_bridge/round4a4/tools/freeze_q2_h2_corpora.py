#!/usr/bin/env python3
"""Freeze disjoint Round4A4 Q2, H2, and F corpora with tokenizer evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer


Q2_NEW = (
    ("q2_en_04", "english_factual", "Which planet has the shortest year in our solar system?"),
    ("q2_zh_04", "chinese_factual", "中国最大的淡水湖通常被认为是哪一个？"),
    ("q2_math_02", "math_numeric", "A shop discounts 240 dollars by 15 percent. What is the final price?"),
    ("q2_math_03", "math_numeric", "Find the greatest common divisor of 84 and 126."),
    ("q2_json_02", "structured_json", "Return only JSON with keys unit, value, and valid for Celsius, 23, true."),
    ("q2_json_03", "structured_json", "Return only a JSON array of objects mapping a to 1 and b to 2."),
    ("q2_code_02", "code_reasoning", "Write Python code that counts the frequency of each word in a list."),
    ("q2_code_03", "code_reasoning", "Explain why binary search requires a sorted sequence."),
    ("q2_code_04", "code_reasoning", "Give pseudocode for breadth-first traversal of a graph."),
)

H2 = (
    ("h2_en_01", "english_factual", "What gas makes up most of Earth's atmosphere?"),
    ("h2_en_02", "english_factual", "Which country contains the ancient city of Petra?"),
    ("h2_en_03", "english_factual", "Who painted The Starry Night?"),
    ("h2_en_04", "english_factual", "What instrument measures atmospheric pressure?"),
    ("h2_zh_01", "chinese_factual", "黄河最终流入哪个海？"),
    ("h2_zh_02", "chinese_factual", "中国传统二十四节气中的第一个节气是什么？"),
    ("h2_zh_03", "chinese_factual", "秦始皇统一六国后建立了哪个朝代？"),
    ("h2_zh_04", "chinese_factual", "太阳系中体积最大的行星是哪一颗？"),
    ("h2_math_01", "math_numeric", "If 5 notebooks cost 42.5 dollars, what is the cost of 8 notebooks?"),
    ("h2_math_02", "math_numeric", "Solve 3(x + 4) = 27 and report x."),
    ("h2_math_03", "math_numeric", "A right triangle has legs 9 and 12. What is its hypotenuse?"),
    ("h2_json_01", "structured_json", "Return only JSON for a book titled Dune with year 1965 and available true."),
    ("h2_json_02", "structured_json", "Return only a JSON object mapping red, green, and blue to their RGB primaries."),
    ("h2_code_01", "code_reasoning", "Write a Python function that merges two already sorted lists."),
    ("h2_code_02", "code_reasoning", "Explain the time complexity of inserting into a hash table."),
    ("h2_code_03", "code_reasoning", "Give pseudocode to find the maximum depth of a binary tree."),
)

F = (
    ("f_en_01", "english_factual", "Why does the Moon show phases when viewed from Earth?"),
    ("f_zh_01", "chinese_factual", "请简要说明都江堰的主要作用。"),
    ("f_math_01", "math_numeric", "A tank is three quarters full at 180 liters. What is its capacity?"),
    ("f_json_01", "structured_json", "Return only JSON listing Monday and Tuesday with indices 1 and 2."),
    ("f_code_01", "code_reasoning", "Write a function that removes duplicates while preserving list order."),
    ("f_code_02", "code_reasoning", "Explain the difference between a queue and a stack with one example."),
)

Q_CATEGORY_MAP = {
    "v_en_01": "english_factual", "v_en_02": "english_factual", "v_en_03": "english_factual",
    "v_zh_01": "chinese_factual", "v_zh_02": "chinese_factual", "v_zh_03": "chinese_factual",
    "v_struct_01": "math_numeric", "v_struct_02": "structured_json", "v_struct_03": "code_reasoning",
}


def canonical_sha(payload: dict) -> str:
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def rows_from_specs(specs: tuple[tuple[str, str, str], ...]) -> list[dict]:
    return [{"id": key, "category": category, "text": text} for key, category, text in specs]


def finalize(name: str, prompts: list[dict], model_dir: Path, tokenizer) -> dict:
    rows = []
    for prompt in prompts:
        row = dict(prompt)
        row["input_ids"] = tokenizer.encode(row["text"], add_special_tokens=False)
        rows.append(row)
    payload = {
        "schema_version": 1,
        "corpus": name,
        "purpose": {"Q2": "pairwise_bound_derivation", "H2": "independent_heldout", "F": "free_generation"}[name],
        "model_dir": str(model_dir),
        "tokenizer_class": tokenizer.__class__.__name__,
        "tokenizer_revision": "DeepSeek-V2-Lite-604d5664 frozen local artifact",
        "protocol": {"teacher_forced_positions": 64, "candidate_top_k": 32, "temperature": 0, "seed": 0},
        "count": len(rows),
        "category_counts": dict(sorted(Counter(row["category"] for row in rows).items())),
        "prompts": rows,
    }
    payload["sha256"] = canonical_sha(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--old-q", type=Path, required=True)
    parser.add_argument("--exclude", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    old_q = json.loads(args.old_q.read_text())
    q_rows = [
        {"id": row["id"], "category": Q_CATEGORY_MAP[row["id"]], "text": row["text"]}
        for row in old_q["prompts"]
    ] + rows_from_specs(Q2_NEW)
    corpora = {
        "Q2": finalize("Q2", q_rows, args.model_dir, tokenizer),
        "H2": finalize("H2", rows_from_specs(H2), args.model_dir, tokenizer),
        "F": finalize("F", rows_from_specs(F), args.model_dir, tokenizer),
    }
    excluded = set()
    for path in args.exclude:
        excluded.update(row["text"] for row in json.loads(path.read_text())["prompts"])
    seen: set[str] = set()
    old_q_texts = {row["text"] for row in old_q["prompts"]}
    for name, corpus in corpora.items():
        texts = {row["text"] for row in corpus["prompts"]}
        if len(texts) != corpus["count"] or texts & seen:
            raise RuntimeError(f"duplicate prompt in or before {name}")
        allowed = old_q_texts if name == "Q2" else set()
        overlap = (texts & excluded) - allowed
        if overlap:
            raise RuntimeError(f"{name} overlaps excluded corpora: {sorted(overlap)}")
        seen.update(texts)
    required = {
        "Q2": {"english_factual": 4, "chinese_factual": 4, "math_numeric": 3, "structured_json": 3, "code_reasoning": 4},
        "H2": {"english_factual": 4, "chinese_factual": 4, "math_numeric": 3, "structured_json": 2, "code_reasoning": 3},
    }
    for name, minima in required.items():
        counts = corpora[name]["category_counts"]
        if any(counts.get(category, 0) < minimum for category, minimum in minima.items()):
            raise RuntimeError(f"{name} category minimum failed: {counts}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, corpus in corpora.items():
        (args.output_dir / f"{name.lower()}.json").write_text(
            json.dumps(corpus, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
    print(json.dumps({name: corpus["sha256"] for name, corpus in corpora.items()}, indent=2))


if __name__ == "__main__":
    main()
