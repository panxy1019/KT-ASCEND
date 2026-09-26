#!/usr/bin/env python3
"""Run one six-generation Round11 final HFINAL reproduction block."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

ROLE_VERSIONS = {
    ("BASELINE", "A"): 147,
    ("FINAL", "A"): 148,
    ("FINAL", "B"): 149,
    ("BASELINE", "B"): 150,
}


def _save(path: Path, value: object) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def _request(rid: str, ids: list[int], maximum: int, *, stream: bool, ignore_eos: bool) -> dict:
    return {
        "rid": rid,
        "input_ids": list(ids),
        "stream": stream,
        "sampling_params": {
            "temperature": 0,
            "top_p": 1,
            "sampling_seed": 20260920,
            "max_new_tokens": maximum,
            "ignore_eos": ignore_eos,
        },
        "return_logprob": True,
        "logprob_start_len": -1,
    }


def _row(fixture: dict, *, version: int, mode: str, block: str, index: int,
         case_id: str, phase: str, maximum: int, decode_tokens: int | None = None) -> dict:
    case = fixture["cases"][case_id]
    stream = phase in {"performance", "cancel"}
    rid = f"r11-t6-v{version}-{mode.lower()}-{block.lower()}-{index:02d}-{case_id}-{phase}"
    return {
        "index": index,
        "mode": mode,
        "block": block,
        "case_id": case_id,
        "phase": phase,
        "decode_tokens": decode_tokens,
        "validator": copy.deepcopy(case["validator"]),
        "request": _request(
            rid,
            case["input_ids"],
            maximum,
            stream=stream,
            ignore_eos=phase == "performance",
        ),
    }


def build_schedule(fixture: dict, *, mode: str, block: str, version: int) -> list[dict]:
    if ROLE_VERSIONS.get((mode, block)) != version:
        raise ValueError("Round11 T6 role/version identity mismatch")
    definitions: list[tuple[str, str, int, int | None]] = [
        ("canary_math391", "warmup", 32, None),
        ("edge128", "performance", 32, 32),
        ("edge128", "performance", 64, 64),
        ("edge128", "performance", 128, 128),
    ]
    if (mode, block) in {("BASELINE", "A"), ("FINAL", "B")}:
        definitions.extend(
            (("cancel128", "cancel", 8, None), ("canary_math391", "reuse", 32, None))
        )
    elif (mode, block) == ("FINAL", "A"):
        definitions.extend(
            (("canary_marker", "canary", 32, None), ("canary_json", "canary", 32, None))
        )
    else:
        definitions.extend(
            (("canary_quote", "canary", 32, None), ("canary_transform", "canary", 32, None))
        )
    return [
        _row(
            fixture,
            version=version,
            mode=mode,
            block=block,
            index=index,
            case_id=case_id,
            phase=phase,
            maximum=maximum,
            decode_tokens=decode_tokens,
        )
        for index, (case_id, phase, maximum, decode_tokens) in enumerate(definitions, 1)
    ]


def _path(row: dict, out: Path, *, protocol, t1) -> dict:
    return protocol.validate_hybrid_path(
        row,
        t1.read_jsonl(out / "hybrid-path.jsonl"),
        t1.read_jsonl(out / "layerwise-events.jsonl"),
        t1.read_jsonl(out / "streaming-events.jsonl"),
    )


def run(mode: str, block: str, version: int, out: Path, fixture_path: Path,
        *, post_fn=None) -> dict:
    import round10_t1_driver as t1
    import round10_t1_protocol as protocol
    from round11_t4_driver import consume_stream

    if post_fn is None:
        post_fn = t1.post
    out = Path(out)
    result_path = out / "round11-t6-raw.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    fixture_path = Path(fixture_path)
    fixture = json.loads(fixture_path.read_text())
    protocol.validate_fixture(fixture)
    schedule = build_schedule(fixture, mode=mode, block=block, version=version)
    report = {
        "schema": 1,
        "status": "RUNNING",
        "mode": mode,
        "block": block,
        "version": version,
        "runtime_identity": "ROUND11_HFINAL_EQUALS_V107_ORIGINAL_PROVIDERS",
        "fixture_sha256": hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
        "profiler": "OFF",
        "generation_budget": 6,
        "generation_attempts": 0,
        "abort_controls": 0,
        "automatic_retries": 0,
        "results": [],
        "cancel_receipts": [],
    }
    pending_cancel = None
    try:
        for row in schedule:
            report["generation_attempts"] += 1
            started = time.monotonic_ns()
            if row["phase"] == "cancel":
                report["abort_controls"] += 1
                pending_cancel = t1.run_cancel(row, out, post_fn=post_fn)
                report["results"].append(
                    {
                        "index": row["index"],
                        "case_id": row["case_id"],
                        "phase": row["phase"],
                        "request_id": row["request"]["rid"],
                        "e2e_seconds": (time.monotonic_ns() - started) / 1e9,
                        "cancel_status": pending_cancel["status"],
                    }
                )
                _save(result_path, report)
                continue

            if row["phase"] == "performance":
                with post_fn("/generate", row["request"], 1800) as response:
                    stream = consume_stream(response, row["decode_tokens"])
                if stream["output_ids"][:2] != [3897, 248046]:
                    raise RuntimeError("T6 forced Decode changed the frozen ACK prefix")
                result = {
                    "index": row["index"],
                    "case_id": row["case_id"],
                    "phase": row["phase"],
                    "request_id": row["request"]["rid"],
                    "input_tokens": len(row["request"]["input_ids"]),
                    "decode_tokens": row["decode_tokens"],
                    "e2e_seconds": (time.monotonic_ns() - started) / 1e9,
                    "quality_pass": True,
                    **stream,
                    "path": _path(row, out, protocol=protocol, t1=t1),
                }
            else:
                with post_fn("/generate", row["request"], 1800) as response:
                    value = json.load(response)
                validated = protocol.validate_response(row, value)
                result = {
                    "index": row["index"],
                    "case_id": row["case_id"],
                    "phase": row["phase"],
                    "request_id": row["request"]["rid"],
                    "e2e_seconds": (time.monotonic_ns() - started) / 1e9,
                    "validated": validated,
                    "output_ids": validated["output_ids"],
                    "path": _path(row, out, protocol=protocol, t1=t1),
                    "quality_pass": True,
                }
                if row["phase"] == "reuse":
                    if pending_cancel is None:
                        raise RuntimeError("T6 reuse is missing its cancellation")
                    pending_cancel["reuse"] = {
                        "request_id": row["request"]["rid"],
                        "validator_passed": True,
                        "output_ids": validated["output_ids"],
                    }
                    pending_cancel["status"] = "PASS_HYBRID_MIDPREFILL_CANCEL_AND_REUSE"
                    protocol.validate_cancel_receipt(pending_cancel)
                    receipt = out / "round11-t6-cancel-receipt.json"
                    _save(receipt, pending_cancel)
                    report["cancel_receipts"].append(receipt.name)
                    report["results"][-1]["cancel_status"] = pending_cancel["status"]
                    pending_cancel = None
            report["results"].append(result)
            _save(result_path, report)

        expected_aborts = int((mode, block) in {("BASELINE", "A"), ("FINAL", "B")})
        if (
            report["generation_attempts"] != 6
            or len(report["results"]) != 6
            or report["abort_controls"] != expected_aborts
            or pending_cancel is not None
            or len(report["cancel_receipts"]) != expected_aborts
        ):
            raise RuntimeError("T6 schedule or cancellation coverage is incomplete")
        report["status"] = "PASS_ROUND11_T6_DRIVER_RAW"
        _save(result_path, report)
        return report
    except BaseException as error:
        report["status"] = "FAIL_ROUND11_T6_DRIVER"
        report["error"] = f"{type(error).__name__}: {error}"
        _save(result_path, report)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("BASELINE", "FINAL"), required=True)
    parser.add_argument("--block", choices=("A", "B"), required=True)
    parser.add_argument("--version", type=int, choices=tuple(ROLE_VERSIONS.values()), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.mode, args.block, args.version, args.out, args.fixture)
    print(result["status"], args.mode, args.block, flush=True)


if __name__ == "__main__":
    main()
