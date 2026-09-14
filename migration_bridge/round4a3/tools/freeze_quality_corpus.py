#!/usr/bin/env python3
"""Freeze deterministic GSM8K and C-Eval quality subsets from local parquet."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import duckdb


CEVAL_SUBJECTS = (
    "chinese_language_and_literature",
    "high_school_history",
    "high_school_geography",
    "basic_medicine",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as reader:
        for chunk in iter(lambda: reader.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gsm8k", type=Path, required=True)
    parser.add_argument("--ceval-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    gsm_rows = duckdb.sql(f"select question, answer from read_parquet('{args.gsm8k}') limit 32").fetchall()
    gsm = []
    for index, (question, answer) in enumerate(gsm_rows):
        match = re.search(r"####\s*([^\n]+)", answer)
        if match is None:
            raise RuntimeError(f"missing GSM8K final answer at row {index}")
        gsm.append(
            {
                "id": f"gsm8k_test_{index:04d}",
                "question": question,
                "reference_answer": match.group(1).strip().replace(",", ""),
                "prompt": question + "\nSolve the problem step by step. End with: Final answer: <number>",
            }
        )
    ceval = []
    source_files = []
    for subject in CEVAL_SUBJECTS:
        path = args.ceval_root / subject / "val-00000-of-00001.parquet"
        source_files.append({"subject": subject, "path": str(path), "sha256": sha256(path)})
        rows = duckdb.sql(f"select id, question, A, B, C, D, answer from read_parquet('{path}') order by id limit 8").fetchall()
        for row_id, question, a, b, c, d, answer in rows:
            prompt = f"{question}\nA. {a}\nB. {b}\nC. {c}\nD. {d}\n请只输出正确选项的字母。"
            ceval.append({"id": f"ceval_{subject}_{row_id}", "subject": subject, "question": question, "choices": {"A": a, "B": b, "C": c, "D": d}, "reference_answer": answer, "prompt": prompt})
    payload = {
        "schema_version": 1,
        "purpose": "independent_downstream_quality",
        "selection": "first 32 rows in frozen dataset order; C-Eval first 8 IDs per frozen subject",
        "protocol": {"temperature": 0, "seed": 0, "gsm8k_max_new_tokens": 256, "ceval_max_new_tokens": 8},
        "benchmarks": {
            "gsm8k": {
                "repository": "openai/gsm8k",
                "revision": "740312add88f781978c0658806c59bc2815b9866",
                "config": "main",
                "split": "test",
                "source_sha256": sha256(args.gsm8k),
                "license": "MIT",
                "count": len(gsm),
                "samples": gsm,
            },
            "ceval": {
                "repository": "ceval/ceval-exam",
                "revision": "617524a00b307ff6f9933702f724131fe12ca7ce",
                "split": "val",
                "subjects": list(CEVAL_SUBJECTS),
                "source_files": source_files,
                "license": "CC-BY-NC-SA-4.0",
                "count": len(ceval),
                "samples": ceval,
            },
        },
    }
    payload["sha256"] = canonical_sha(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(payload["sha256"])


if __name__ == "__main__":
    main()
