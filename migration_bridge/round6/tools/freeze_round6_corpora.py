#!/usr/bin/env python3
"""Freeze disjoint Round 6 Deferred qualification prompt corpora.

This tool freezes prompts and tokenizer input IDs before placement selection or
Deferred output inspection.  All-NPU baseline histories are captured later and
bound to these immutable prompt-corpus hashes in separate evidence manifests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer


MODEL_REVISION = "604d5664dddd88a0433dbae533b7fe9472482de0"
TOKENIZER_SHA256 = "41f3bf64213da8c012d8bd0871a58a1fdf70463e8f08f110ddbb1082f529f669"


S6 = (
    ("s6_en_01", "english_factual", "Explain why ocean tides change during a lunar month."),
    ("s6_en_02", "english_factual", "Name the layers of Earth's atmosphere from lowest to highest."),
    ("s6_en_03", "english_factual", "Describe how a compass indicates geographic direction."),
    ("s6_zh_01", "chinese_factual", "简要说明长江三峡的地理位置和主要作用。"),
    ("s6_zh_02", "chinese_factual", "中国古代四大发明分别是什么？"),
    ("s6_zh_03", "chinese_factual", "为什么高海拔地区水的沸点较低？"),
    ("s6_math_01", "math_numeric", "Compute the sum of the first twelve positive odd integers."),
    ("s6_math_02", "math_numeric", "A rectangle has perimeter 54 and length 16. Find its width."),
    ("s6_json_01", "structured_json", "Return only JSON describing three traffic-light colors and their actions."),
    ("s6_json_02", "structured_json", "Return only a JSON object with the squares of integers 2 through 6."),
    ("s6_code_01", "code_reasoning", "Write pseudocode for detecting a cycle in a singly linked list."),
    ("s6_code_02", "code_reasoning", "Explain when a stable sorting algorithm matters, with one example."),
)


QD = (
    ("qd_en_01", "english_factual", "Why do coastal regions often have milder temperatures than inland regions?"),
    ("qd_en_02", "english_factual", "What physical process creates a rainbow after rain?"),
    ("qd_en_03", "english_factual", "Explain the role of platelets in the human body."),
    ("qd_en_04", "english_factual", "Which properties distinguish a comet from an asteroid?"),
    ("qd_zh_01", "chinese_factual", "请说明季风形成的主要原因。"),
    ("qd_zh_02", "chinese_factual", "简述京杭大运河在历史上的作用。"),
    ("qd_zh_03", "chinese_factual", "中国的暖温带与亚热带通常以哪条地理界线划分？"),
    ("qd_zh_04", "chinese_factual", "为什么金属通常具有良好的导电性？"),
    ("qd_math_01", "math_numeric", "A train travels 315 kilometers in 4.5 hours. What is its average speed?"),
    ("qd_math_02", "math_numeric", "Find the least common multiple of 36, 48, and 60."),
    ("qd_math_03", "math_numeric", "A circle has radius 7. Express its area using pi."),
    ("qd_json_01", "structured_json", "Return only JSON for two sensors with names, units, and boolean active states."),
    ("qd_json_02", "structured_json", "Return only a JSON array containing the months in the first quarter."),
    ("qd_code_01", "code_reasoning", "Write Python code to group words by their first letter while preserving order."),
    ("qd_code_02", "code_reasoning", "Explain the invariant maintained by insertion sort."),
    ("qd_code_03", "code_reasoning", "Give pseudocode that finds the first non-repeating character in a string."),
)


HD = (
    ("hd_en_01", "english_factual", "How does a barometer help forecast short-term weather changes?"),
    ("hd_en_02", "english_factual", "Why are leap years needed in the Gregorian calendar?"),
    ("hd_en_03", "english_factual", "Describe the difference between weathering and erosion."),
    ("hd_en_04", "english_factual", "What causes the apparent retrograde motion of planets?"),
    ("hd_zh_01", "chinese_factual", "请解释喀斯特地貌形成的基本过程。"),
    ("hd_zh_02", "chinese_factual", "郑和下西洋发生在哪个朝代，主要目的是什么？"),
    ("hd_zh_03", "chinese_factual", "为什么冬季室内玻璃上有时会出现水珠？"),
    ("hd_zh_04", "chinese_factual", "简要说明黑土适合农业生产的原因。"),
    ("hd_math_01", "math_numeric", "A mixture contains water and syrup in a 7 to 3 ratio. How much syrup is in 250 ml?"),
    ("hd_math_02", "math_numeric", "Solve 5y minus 17 equals 3y plus 9."),
    ("hd_math_03", "math_numeric", "The angles of a triangle are in ratio 2:3:4. Find all three angles."),
    ("hd_json_01", "structured_json", "Return only JSON mapping north, east, south, and west to degree bearings."),
    ("hd_json_02", "structured_json", "Return only a JSON object for a library book with nested author and loan fields."),
    ("hd_code_01", "code_reasoning", "Write a function that rotates a list right by k positions without losing elements."),
    ("hd_code_02", "code_reasoning", "Explain why a breadth-first search finds shortest paths in an unweighted graph."),
    ("hd_code_03", "code_reasoning", "Give pseudocode to validate balanced parentheses of three bracket types."),
)


FD = (
    ("fd_en_01", "english_factual", "Explain how migratory birds navigate over long distances."),
    ("fd_en_02", "english_factual", "Why does iron rust faster in salty water?"),
    ("fd_zh_01", "chinese_factual", "请简述丝绸之路促进文化交流的方式。"),
    ("fd_zh_02", "chinese_factual", "为什么月球表面昼夜温差很大？"),
    ("fd_math_01", "math_numeric", "A box contains 5 red, 7 blue, and 8 green balls. What fraction are blue?"),
    ("fd_json_01", "structured_json", "Return only JSON listing four seasons with ordinal numbers."),
    ("fd_code_01", "code_reasoning", "Write pseudocode for merging overlapping numeric intervals."),
    ("fd_code_02", "code_reasoning", "Compare recursion and iteration for computing a factorial."),
)


SPECS = {"S6": S6, "QD": QD, "HD": HD, "FD": FD}
PURPOSE = {
    "S6": "defer0_route_stress_and_placement_selection",
    "QD": "deferred_numerical_contract_derivation",
    "HD": "independent_deferred_numerical_heldout",
    "FD": "deferred_free_generation_validation",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as reader:
        for chunk in iter(lambda: reader.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: dict) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_excluded_texts(paths: list[Path]) -> set[str]:
    texts: set[str] = set()
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload.get("prompts", []):
            texts.add(row["text"])
    return texts


def freeze_corpus(name: str, specs, model_dir: Path, tokenizer) -> dict:
    prompts = [
        {
            "id": prompt_id,
            "category": category,
            "text": text,
            "input_ids": tokenizer.encode(text, add_special_tokens=False),
        }
        for prompt_id, category, text in specs
    ]
    payload = {
        "schema_version": 1,
        "round": "round6",
        "corpus": name,
        "purpose": PURPOSE[name],
        "model": "deepseek-ai/DeepSeek-V2-Lite",
        "model_revision": MODEL_REVISION,
        "model_dir": str(model_dir),
        "tokenizer_class": tokenizer.__class__.__name__,
        "tokenizer_sha256": TOKENIZER_SHA256,
        "protocol": {
            "teacher_forced_positions": 64,
            "candidate_top_k": 32,
            "temperature": 0,
            "seed": 0,
            "baseline_history": "captured_after_prompt_freeze_and_bound_by_manifest",
        },
        "count": len(prompts),
        "category_counts": dict(
            sorted(Counter(row["category"] for row in prompts).items())
        ),
        "prompts": prompts,
    }
    payload["sha256"] = canonical_sha256(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--exclude", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    actual_tokenizer_sha = file_sha256(args.model_dir / "tokenizer.json")
    if actual_tokenizer_sha != TOKENIZER_SHA256:
        raise RuntimeError(
            "tokenizer drift: "
            f"expected {TOKENIZER_SHA256}, got {actual_tokenizer_sha}"
        )

    excluded = load_excluded_texts(args.exclude)
    all_ids: set[str] = set()
    all_texts: set[str] = set()
    for name, specs in SPECS.items():
        ids = {row[0] for row in specs}
        texts = {row[2] for row in specs}
        if len(ids) != len(specs) or len(texts) != len(specs):
            raise RuntimeError(f"duplicates inside {name}")
        if ids & all_ids or texts & all_texts:
            raise RuntimeError(f"cross-corpus duplicate in {name}")
        overlap = texts & excluded
        if overlap:
            raise RuntimeError(f"{name} overlaps prior frozen corpus: {sorted(overlap)}")
        all_ids.update(ids)
        all_texts.update(texts)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    corpora = {
        name: freeze_corpus(name, specs, args.model_dir, tokenizer)
        for name, specs in SPECS.items()
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in corpora.items():
        output = args.output_dir / f"{name.lower()}.json"
        output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(
        json.dumps(
            {name: payload["sha256"] for name, payload in corpora.items()},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
