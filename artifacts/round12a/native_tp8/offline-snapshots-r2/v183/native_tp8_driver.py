#!/usr/bin/env python3
"""Run one frozen native-SGLang TP8/old-QSA W1 process replica."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time


ROLES = {180: "A", 181: "B", 182: "B", 183: "A"}
QSA_LAYERS = list(range(3, 48, 4))
GDN_LAYERS = [layer for layer in range(48) if layer not in QSA_LAYERS]
EXPECTED_OUTPUT_IDS = [3897, 248046]
REQUEST_WALL_LIMIT_SECONDS = 1800


def _save(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def _request(rid: str, ids: list[int]) -> dict:
    return {
        "rid": rid,
        "input_ids": list(ids),
        "stream": True,
        "sampling_params": {
            "temperature": 0,
            "top_p": 1,
            "sampling_seed": 20260920,
            "max_new_tokens": 2,
            "ignore_eos": True,
        },
        "return_logprob": True,
        "logprob_start_len": -1,
    }


def build_schedule(fixture: dict, *, version: int, block: str) -> list[dict]:
    if ROLES.get(version) != block:
        raise ValueError("native TP8 role/version mismatch")
    ids = fixture.get("cases", {}).get("edge64", {}).get("input_ids")
    if not isinstance(ids, list) or len(ids) != 64 or any(type(x) is not int for x in ids):
        raise ValueError("native TP8 requires the frozen exact 64-token fixture")
    rows = []
    for phase, repeat in (("warmup", 0), ("measurement", 1), ("measurement", 2), ("measurement", 3)):
        rid = f"r12a-native-tp8-v{version}-{block.lower()}-w1-{phase}-{repeat}"
        rows.append({
            "index": len(rows) + 1,
            "phase": phase,
            "repeat": repeat,
            "workload": "W1",
            "input_tokens": 64,
            "decode_tokens": 2,
            "request": _request(rid, ids),
        })
    return rows


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise RuntimeError(f"unterminated rank evidence: {path.name}")
    return [json.loads(row) for row in raw.splitlines()]


def read_rank_evidence(out: Path) -> dict[int, list[dict]]:
    return {rank: _read_jsonl(out / f"native-tp8-rank{rank}.jsonl") for rank in range(8)}


def validate_rank_evidence(events: dict[int, list[dict]]) -> dict:
    if set(events) != set(range(8)):
        raise RuntimeError("native TP8 evidence does not cover all ranks")
    physical = {}
    qsa = {}
    gdn = {}
    for rank in range(8):
        rows = events[rank]
        physical_rows = [r for r in rows if r.get("status") == "NATIVE_TP8_LAYER_PHYSICAL"]
        qsa_rows = [r for r in rows if r.get("status") == "NATIVE_TP8_OLD_QSA_HIT"]
        gdn_rows = [r for r in rows if r.get("status") == "NATIVE_TP8_GDN_HIT"]
        if sorted(r.get("layer") for r in physical_rows) != list(range(48)):
            raise RuntimeError(f"rank {rank} lacks all-layer physical evidence")
        if any(
            r.get("rank") != rank or r.get("w13_device") != "npu"
            or r.get("w2_device") != "npu" or r.get("dtype") != "torch.bfloat16"
            or int(r.get("w13_bytes", 1)) <= 0 or int(r.get("w2_bytes", 1)) <= 0
            for r in physical_rows
        ):
            raise RuntimeError(f"rank {rank} has invalid expert physical evidence")
        if sorted(r.get("layer") for r in qsa_rows) != QSA_LAYERS:
            raise RuntimeError(f"rank {rank} lacks all old-QSA layer hits")
        if any(
            r.get("rank") != rank or r.get("query_heads") != 3 or r.get("kv_heads") != 1
            or int(r.get("query_tokens", 0)) <= 0
            for r in qsa_rows
        ):
            raise RuntimeError(f"rank {rank} has invalid old-QSA local geometry")
        if sorted(r.get("layer") for r in gdn_rows) != GDN_LAYERS:
            raise RuntimeError(f"rank {rank} lacks all TP-local GDN hits")
        if any(
            r.get("rank") != rank or r.get("key_heads") != 2
            or r.get("value_heads") != 6 or r.get("z_heads") != 6
            or r.get("conv_state_device") != "npu"
            or r.get("recurrent_state_device") != "npu"
            or not r.get("conv_state_shape") or not r.get("recurrent_state_shape")
            for r in gdn_rows
        ):
            raise RuntimeError(f"rank {rank} has invalid TP-local GDN geometry")
        physical[rank] = {
            "layers": 48,
            "expert_bytes": sum(int(r["w13_bytes"]) + int(r["w2_bytes"]) for r in physical_rows),
        }
        qsa[rank] = {"layers": QSA_LAYERS, "query_heads": 3, "kv_heads": 1}
        gdn[rank] = {"layers": GDN_LAYERS, "key_heads": 2, "value_heads": 6}
    return {
        "ranks": list(range(8)), "physical": physical, "old_qsa": qsa,
        "gdn_tp_local": gdn,
    }


def validate_ple(out: Path, request_id: str) -> dict:
    summary = {}
    for rank in range(8):
        rows = _read_jsonl(out / f"ple-async-events-rank{rank}.jsonl")
        if sum(row.get("status") == "PLE_ASYNC_BOUND" for row in rows) != 1:
            raise RuntimeError(f"rank {rank} PLE ASYNC binding missing")
        drained = [
            row for row in rows
            if row.get("status") == "PLE_BATCH_DRAINED" and row.get("request_id") == request_id
        ]
        if not drained or not any(row.get("async_engaged") is True for row in drained):
            raise RuntimeError(f"rank {rank} request did not engage PLE ASYNC")
        summary[rank] = {
            "drained_batches": len(drained),
            "async_batches": sum(row.get("async_engaged") is True for row in drained),
        }
    return summary


def _validate_queue(path: Path) -> str:
    raw = path.read_bytes()
    doc = json.loads(raw)
    if (
        doc.get("status") != "FROZEN_ROUND12A_ABBA_QUEUE"
        or doc.get("workloads") != ["W1"]
        or doc.get("workload_definitions") != {
            "W1": {"case_id": "edge64", "input_tokens": 64, "decode_tokens": 2}
        }
        or doc.get("expected_output_ids") != {"W1": EXPECTED_OUTPUT_IDS}
        or doc.get("request_wall_limit_seconds") != REQUEST_WALL_LIMIT_SECONDS
    ):
        raise ValueError("native TP8 queue differs from Round12a W1")
    return hashlib.sha256(raw).hexdigest()


def run(version: int, block: str, out: Path, fixture_path: Path, queue_path: Path, *, post_fn=None) -> dict:
    from round11_t4_driver import consume_stream, post

    post_fn = post if post_fn is None else post_fn
    out = Path(out)
    result_path = out / "native-tp8-raw.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    fixture = json.loads(Path(fixture_path).read_text())
    schedule = build_schedule(fixture, version=version, block=block)
    runtime = json.loads((out / "native-tp8-runtime.json").read_text())
    if (
        runtime.get("status") != "PASS_ROUND12A_NATIVE_TP8_SERVER_PREFLIGHT"
        or runtime.get("tp_size") != 8
        or runtime.get("qsa_mode") != "QUALIFIED_ORIGINAL"
        or runtime.get("cpu_offload_gb") != 0
        or runtime.get("kt_weight_path") is not None
    ):
        raise RuntimeError("native TP8 server identity receipt failed")
    report = {
        "schema": 1, "status": "RUNNING", "stage": "ROUND12A_NATIVE_TP8_OLD_QSA_ABBA",
        "version": version, "mode": "NATIVE_TP8", "block": block,
        "profiler": "OFF", "timing_primary": "CLIENT_UNPROFILED_E2E",
        "fixture_sha256": hashlib.sha256(Path(fixture_path).read_bytes()).hexdigest(),
        "queue_sha256": _validate_queue(Path(queue_path)),
        "generation_budget": 4, "generation_attempts": 0, "abort_controls": 0,
        "automatic_retries": 0, "runtime": runtime, "results": [],
    }
    try:
        for row in schedule:
            report["generation_attempts"] += 1
            started = time.monotonic_ns()
            with post_fn("/generate", row["request"], REQUEST_WALL_LIMIT_SECONDS) as response:
                stream = consume_stream(response, 2)
            ended = time.monotonic_ns()
            arrivals = stream["token_arrival_monotonic_ns"]
            metrics = {
                "ttft_seconds": (arrivals[0] - started) / 1e9,
                "e2e_seconds": (ended - started) / 1e9,
                "tpot_seconds": (arrivals[1] - arrivals[0]) / 1e9,
            }
            if metrics["e2e_seconds"] >= REQUEST_WALL_LIMIT_SECONDS:
                raise RuntimeError("native TP8 request exceeded the frozen wall limit")
            if stream["output_ids"] != EXPECTED_OUTPUT_IDS or stream.get("finite_logprobs") is not True:
                raise RuntimeError("native TP8 output differs from the frozen quality contract")
            rank_receipt = validate_rank_evidence(read_rank_evidence(out))
            ple = validate_ple(out, row["request"]["rid"])
            report["results"].append({
                "index": row["index"], "workload": "W1", "phase": row["phase"],
                "repeat": row["repeat"], "request_id": row["request"]["rid"],
                "input_tokens": 64, "decode_tokens": 2, **metrics,
                "quality_pass": True, "output_ids": stream["output_ids"],
                "finite_logprobs": True, "finish_reason": stream["finish_reason"],
                "completion_tokens": stream["completion_tokens"],
                "path": "SGLANG_NATIVE_TP8_BF16", "qsa": "QUALIFIED_ORIGINAL",
                "ple": ple,
            })
            report["rank_evidence"] = rank_receipt
            _save(result_path, report)
        if report["generation_attempts"] != 4 or len(report["results"]) != 4:
            raise RuntimeError("native TP8 schedule incomplete")
        report["status"] = "PASS_ROUND12A_NATIVE_TP8_DRIVER_RAW"
        _save(result_path, report)
        print(report["status"], version, block, flush=True)
        return report
    except BaseException as error:
        report.update(status="FAIL", error=repr(error))
        _save(result_path, report)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--block", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    args = parser.parse_args()
    run(args.version, args.block, args.out, args.fixture, args.queue)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
