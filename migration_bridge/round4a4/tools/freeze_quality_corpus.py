#!/usr/bin/env python3
"""Freeze pinned 128-example GSM8K and C-Eval quality sets from local parquet."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from decimal import Decimal
from pathlib import Path

import duckdb


CEVAL_SUBJECTS = (
    "basic_medicine",
    "chinese_language_and_literature",
    "college_programming",
    "high_school_geography",
    "high_school_history",
    "high_school_mathematics",
    "law",
    "modern_chinese_history",
)
GSM8K_REVISION = "740312add88f781978c0658806c59bc2815b9866"
CEVAL_REVISION = "617524a00b307ff6f9933702f724131fe12ca7ce"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as reader:
        for chunk in iter(lambda: reader.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: dict) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def format_decimal(value: Decimal) -> str:
    normalized = value.normalize()
    if normalized == normalized.to_integral():
        return format(normalized, "f")
    return format(normalized, "f").rstrip("0").rstrip(".")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gsm8k", type=Path, required=True)
    parser.add_argument("--ceval-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    gsm_rows = duckdb.sql(
        f"select question, answer from read_parquet('{args.gsm8k}') limit 128"
    ).fetchall()
    gsm8k = []
    for index, (question, answer) in enumerate(gsm_rows):
        match = re.search(r"####\s*([^\n]+)", answer)
        if match is None:
            raise RuntimeError(f"missing GSM8K final answer at row {index}")
        reference_number = Decimal(match.group(1).strip().replace(",", ""))
        candidate_numbers = []
        for value in (
            reference_number,
            reference_number + 1,
            reference_number - 1,
            reference_number * 2,
            reference_number + 2,
            reference_number - 2,
        ):
            if value not in candidate_numbers:
                candidate_numbers.append(value)
            if len(candidate_numbers) == 4:
                break
        rng = random.Random(20260830 + index)
        rng.shuffle(candidate_numbers)
        choices = {
            letter: format_decimal(value)
            for letter, value in zip("ABCD", candidate_numbers)
        }
        reference_answer = next(
            letter for letter, value in zip("ABCD", candidate_numbers) if value == reference_number
        )
        choice_text = "\n".join(f"{letter}. {value}" for letter, value in choices.items())
        gsm8k.append(
            {
                "id": f"gsm8k_test_{index:04d}",
                "question": question,
                "source_numeric_answer": format_decimal(reference_number),
                "choices": choices,
                "reference_answer": reference_answer,
                "prompt": question
                + "\n"
                + choice_text
                + "\nChoose the correct answer. Reply with the option letter.",
            }
        )

    ceval_pool = []
    source_files = []
    for subject in CEVAL_SUBJECTS:
        path = args.ceval_root / subject / "val-00000-of-00001.parquet"
        source_files.append({"subject": subject, "sha256": file_sha256(path)})
        rows = duckdb.sql(
            f"select id, question, A, B, C, D, answer from read_parquet('{path}') order by id"
        ).fetchall()
        for row_id, question, a, b, c, d, answer in rows:
            ceval_pool.append(
                {
                    "id": f"ceval_{subject}_{row_id}",
                    "subject": subject,
                    "question": question,
                    "choices": {"A": a, "B": b, "C": c, "D": d},
                    "reference_answer": answer,
                    "prompt": f"{question}\nA. {a}\nB. {b}\nC. {c}\nD. {d}\n请只输出正确选项的字母。",
                }
            )
    ceval = ceval_pool[:128]
    if len(gsm8k) != 128 or len(ceval) != 128:
        raise RuntimeError(f"quality corpus too small: GSM8K={len(gsm8k)} C-Eval={len(ceval)}")

    payload = {
        "schema_version": 1,
        "purpose": "round4a4_independent_downstream_quality",
        "selection": "first 128 frozen GSM8K test rows; first 128 rows from ordered frozen C-Eval subject pool",
        "protocol": {
            "seed": 0,
            "gsm8k_scoring": "deterministic_four_choice_conditional_loglikelihood",
            "ceval_scoring": "conditional_loglikelihood_over_A_B_C_D",
        },
        "benchmarks": {
            "gsm8k": {
                "repository": "openai/gsm8k",
                "revision": GSM8K_REVISION,
                "config": "main",
                "split": "test",
                "source_sha256": file_sha256(args.gsm8k),
                "license": "MIT",
                "count": len(gsm8k),
                "samples": gsm8k,
            },
            "ceval": {
                "repository": "ceval/ceval-exam",
                "revision": CEVAL_REVISION,
                "split": "val",
                "subjects": list(CEVAL_SUBJECTS),
                "source_files": source_files,
                "license": "CC-BY-NC-SA-4.0",
                "count": len(ceval),
                "samples": ceval,
            },
        },
    }
    payload["sha256"] = canonical_sha256(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(payload["sha256"])


if __name__ == "__main__":
    main()
