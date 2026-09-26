#!/usr/bin/env python3
"""Run the one bounded, unprofiled Round11 QSA candidate model process."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

try:
    from .round11_t4_driver import consume_stream, post, read_jsonl, validate_hybrid_path
except ImportError:  # pragma: no cover - remote snapshot entry point
    from round11_t4_driver import consume_stream, post, read_jsonl, validate_hybrid_path


CANARY_OUTPUT_IDS = [18, 24, 16, 248046]
ACK_PREFIX_IDS = [3897, 248046]


def _save(path: Path, value: object) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def _ids(fixture: dict, case_id: str, expected_length: int) -> list[int]:
    values = fixture.get("cases", {}).get(case_id, {}).get("input_ids")
    if (
        not isinstance(values, list)
        or len(values) != expected_length
        or any(type(value) is not int or value < 0 for value in values)
    ):
        raise ValueError(f"QSA model block requires frozen {case_id} input IDs")
    return list(values)


def build_schedule(fixture: dict, *, version: int = 145) -> list[dict]:
    if version not in {145, 146}:
        raise ValueError("unsupported QSA model run identity")
    edge = _ids(fixture, "edge128", 128)
    canary = _ids(fixture, "canary_math391", 27)
    definitions = (
        ("canary_math391", "warmup", canary, 4),
        ("edge128", "measurement", edge, 32),
        ("edge128", "measurement", edge, 64),
        ("canary_math391", "drain", canary, 4),
    )
    rows = []
    for index, (case_id, phase, input_ids, decode_tokens) in enumerate(definitions, 1):
        rid = f"r11-t5-qsa-v{version}-{index:02d}-{case_id}-{phase}-decode{decode_tokens}"
        rows.append(
            {
                "index": index,
                "case_id": case_id,
                "phase": phase,
                "decode_tokens": decode_tokens,
                "request": {
                    "rid": rid,
                    "input_ids": input_ids,
                    "stream": True,
                    "sampling_params": {
                        "temperature": 0,
                        "top_p": 1,
                        "sampling_seed": 20260920,
                        "max_new_tokens": decode_tokens,
                        "ignore_eos": True,
                    },
                    "return_logprob": True,
                    "logprob_start_len": -1,
                },
            }
        )
    return rows


def validate_output(row: dict, output_ids: list[int]) -> bool:
    if row["case_id"] == "canary_math391":
        if output_ids != CANARY_OUTPUT_IDS:
            raise RuntimeError("QSA candidate changed the frozen arithmetic canary")
    elif row["case_id"] == "edge128":
        if output_ids[: len(ACK_PREFIX_IDS)] != ACK_PREFIX_IDS:
            raise RuntimeError("QSA candidate changed the frozen ACK prefix")
    else:  # Defensive: schedule identities are part of the evidence contract.
        raise RuntimeError("QSA candidate schedule is not frozen")
    return True


def run(
    out: Path,
    fixture_path: Path,
    parent_identity: str,
    *,
    version: int = 145,
    post_fn=post,
) -> dict:
    out = Path(out)
    result_path = out / "round11-t5-qsa-model-raw.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    fixture_path = Path(fixture_path)
    fixture = json.loads(fixture_path.read_text())
    schedule = build_schedule(fixture, version=version)
    report = {
        "schema": 1,
        "status": "RUNNING",
        "version": version,
        "candidate": "QUERY_AWARE_VECTORIZED_SPARSE_GQA",
        "parent_identity": parent_identity,
        "fixture_sha256": hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
        "profiler": "OFF",
        "generation_budget": 4,
        "generation_attempts": 0,
        "automatic_retries": 0,
        "results": [],
    }
    try:
        for row in schedule:
            report["generation_attempts"] += 1
            rid = row["request"]["rid"]
            started = time.monotonic_ns()
            with post_fn("/generate", row["request"], 1800) as response:
                stream = consume_stream(response, row["decode_tokens"])
            ended = time.monotonic_ns()
            path = validate_hybrid_path(
                rid,
                read_jsonl(out / "hybrid-path.jsonl"),
                read_jsonl(out / "layerwise-events.jsonl"),
            )
            validate_output(row, stream["output_ids"])
            report["results"].append(
                {
                    "request_id": rid,
                    "case_id": row["case_id"],
                    "phase": row["phase"],
                    "input_tokens": len(row["request"]["input_ids"]),
                    "decode_tokens": row["decode_tokens"],
                    "e2e_seconds": (ended - started) / 1e9,
                    "quality_pass": True,
                    **stream,
                    "path": "HYBRID",
                    "path_evidence": path,
                }
            )
            _save(result_path, report)
        if report["generation_attempts"] != 4 or len(report["results"]) != 4:
            raise RuntimeError("QSA candidate model schedule is incomplete")
        report["status"] = "PASS_ROUND11_T5_QSA_MODEL_DRIVER_RAW"
        _save(result_path, report)
        return report
    except BaseException as error:
        report["status"] = "FAIL_ROUND11_T5_QSA_MODEL_DRIVER"
        report["error"] = f"{type(error).__name__}: {error}"
        _save(result_path, report)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--parent-identity", required=True)
    parser.add_argument("--version", type=int, choices=(145, 146), default=145)
    args = parser.parse_args()
    result = run(args.out, args.fixture, args.parent_identity, version=args.version)
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
