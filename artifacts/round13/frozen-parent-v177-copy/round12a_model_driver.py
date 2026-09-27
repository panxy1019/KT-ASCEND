#!/usr/bin/env python3
"""Run one frozen Round12a promoted-QSA non-KT smoke or KT/non-KT ABBA arm."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time


ROLES = {
    173: ("NON_KT", "SMOKE"),
    174: ("KT", "A"),
    175: ("NON_KT", "A"),
    176: ("NON_KT", "B"),
    177: ("KT", "B"),
}
QSA_LAYERS = list(range(3, 48, 4))
WORKLOADS = {
    "W1": {"case_id": "edge64", "input_tokens": 64, "decode_tokens": 2},
}
SMOKE_DEFINITION = {"case_id": "edge64", "input_tokens": 64, "decode_tokens": 4}
SMOKE_EXPECTED_OUTPUT_IDS = [3897, 248046, 198, 248044]
W1_EXPECTED_OUTPUT_IDS = [3897, 248046]
REQUEST_WALL_LIMIT_SECONDS = 1800


def _save(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def decide_workloads(
    *,
    smoke_ttft_seconds: float,
    smoke_tpot_seconds: float,
    smoke_e2e_seconds: float,
) -> dict:
    """Freeze the smallest request that exercises Prefill and one Decode step.

    v173 established that the four-token non-KT smoke exceeds the request wall
    limit.  Its observed TTFT plus one observed token interval is the only
    evidence-based projection available without spending another model run.
    """
    projected = smoke_ttft_seconds + smoke_tpot_seconds
    return {
        "workloads": ["W1"],
        "projection_formula": "smoke_ttft_seconds + smoke_tpot_seconds",
        "projected_w1_seconds": projected,
        "request_wall_limit_seconds": REQUEST_WALL_LIMIT_SECONDS,
        "source_smoke_exceeded_request_wall_limit": (
            smoke_e2e_seconds >= REQUEST_WALL_LIMIT_SECONDS
        ),
    }


def validate_request_wall(e2e_seconds: float, *, version: int) -> None:
    """Enforce the declared wall limit for every formal ABBA request."""
    if version != 173 and e2e_seconds >= REQUEST_WALL_LIMIT_SECONDS:
        raise RuntimeError(
            f"Round12a request wall-time limit exceeded: "
            f"{e2e_seconds:.6f} >= {REQUEST_WALL_LIMIT_SECONDS}"
        )


def _request(rid: str, ids: list[int], decode_tokens: int) -> dict:
    return {
        "rid": rid,
        "input_ids": list(ids),
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
    }


def build_schedule(
    fixture: dict,
    *,
    version: int,
    mode: str,
    block: str,
    workloads: list[str],
) -> list[dict]:
    if ROLES.get(version) != (mode, block):
        raise ValueError("Round12a role/version mismatch")
    if not workloads or any(workload not in WORKLOADS for workload in workloads):
        raise ValueError("Round12a workload queue is invalid")
    if version == 173:
        if workloads != ["W1"]:
            raise ValueError("Round12a smoke is fixed to W1")
        repeats = [("smoke", 0)]
    else:
        repeats = [
            ("warmup", 0),
            ("measurement", 1),
            ("measurement", 2),
            ("measurement", 3),
        ]
    rows = []
    for workload in workloads:
        definition = SMOKE_DEFINITION if version == 173 else WORKLOADS[workload]
        case = fixture["cases"][definition["case_id"]]
        for phase, repeat in repeats:
            rid = (
                f"r12a-v{version}-{mode.lower()}-{block.lower()}-"
                f"{workload.lower()}-{phase}-{repeat}"
            )
            rows.append(
                {
                    "index": len(rows) + 1,
                    "workload": workload,
                    "case_id": definition["case_id"],
                    "phase": phase,
                    "repeat": repeat,
                    "input_tokens": len(case["input_ids"]),
                    "decode_tokens": definition["decode_tokens"],
                    "request": _request(
                        rid, case["input_ids"], definition["decode_tokens"]
                    ),
                }
            )
    return rows


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def validate_candidate_qsa(request_id: str, events: list[dict]) -> dict:
    rows = [
        row
        for row in events
        if row.get("status") == "CONTEXT_REFERENCE_QSA_DISPATCH"
        and row.get("request_id") == request_id
    ]
    vector = [row for row in rows if row.get("strategy") == "VECTORIZED"]
    row_path = [row for row in rows if row.get("strategy") == "ROW"]
    vector_layers = sorted({row.get("layer") for row in vector})
    row_layers = sorted({row.get("layer") for row in row_path})
    if vector_layers != QSA_LAYERS:
        raise RuntimeError("request did not hit all promoted VECTOR QSA layers")
    if row_layers != QSA_LAYERS:
        raise RuntimeError("Decode did not retain all promoted ROW QSA layers")
    if any(row.get("query_tokens", 0) <= 1 for row in vector):
        raise RuntimeError("promoted VECTOR QSA event has invalid query size")
    return {
        "mode": "APPROVED_CANDIDATE",
        "dispatch_count": len(rows),
        "vectorized_count": len(vector),
        "row_count": len(row_path),
        "vectorized_layers": vector_layers,
        "row_layers": row_layers,
    }


def validate_qsa_binding(events: list[dict]) -> dict:
    rows = [row for row in events if row.get("status") == "CONTEXT_REFERENCE_BOUND"]
    if len(rows) != 1:
        raise RuntimeError("Round12a requires one context binding event")
    binding = rows[0].get("qsa_source_binding", {})
    if binding.get("mode") != "APPROVED_CANDIDATE":
        raise RuntimeError("Round12a did not bind the promoted QSA tuple")
    return binding


def validate_ple(events: list[dict], request_id: str) -> dict:
    if not any(row.get("status") == "PLE_ASYNC_BOUND" for row in events):
        raise RuntimeError("PLE ASYNC binding event missing")
    drained = [
        row
        for row in events
        if row.get("status") == "PLE_BATCH_DRAINED"
        and row.get("request_id") == request_id
    ]
    if not drained or not any(row.get("async_engaged") is True for row in drained):
        raise RuntimeError(f"request {request_id} did not engage qualified PLE ASYNC")
    return {
        "drained_batches": len(drained),
        "async_batches": sum(row.get("async_engaged") is True for row in drained),
    }


def timing_metrics(started_ns: int, ended_ns: int, arrivals: list[int]) -> dict:
    if not arrivals or arrivals[0] < started_ns or ended_ns < arrivals[-1]:
        raise ValueError("stream timing order is invalid")
    return {
        "ttft_seconds": (arrivals[0] - started_ns) / 1e9,
        "e2e_seconds": (ended_ns - started_ns) / 1e9,
        "tpot_seconds": (
            (arrivals[-1] - arrivals[0]) / (len(arrivals) - 1) / 1e9
            if len(arrivals) > 1
            else None
        ),
        "timing_primary": "CLIENT_UNPROFILED_E2E",
    }


def _load_queue(queue_path: Path | None, *, version: int) -> tuple[list[str], str | None]:
    if version == 173:
        if queue_path is not None:
            raise ValueError("smoke must run before an ABBA queue exists")
        return ["W1"], None
    if queue_path is None:
        raise ValueError("ABBA requires a frozen queue")
    raw = queue_path.read_bytes()
    document = json.loads(raw)
    if (
        document.get("schema") != 1
        or document.get("status") != "FROZEN_ROUND12A_ABBA_QUEUE"
        or document.get("source_smoke_version") != 173
        or document.get("workloads") != ["W1"]
        or document.get("workload_definitions") != WORKLOADS
        or document.get("request_wall_limit_seconds")
        != REQUEST_WALL_LIMIT_SECONDS
    ):
        raise ValueError("invalid Round12a ABBA queue")
    return document["workloads"], hashlib.sha256(raw).hexdigest()


def run(
    mode: str,
    block: str,
    version: int,
    out: Path,
    fixture_path: Path,
    queue_path: Path | None,
    *,
    post_fn=None,
) -> dict:
    from round11_t4_driver import consume_stream, post, read_jsonl, validate_hybrid_path

    if post_fn is None:
        post_fn = post
    out = Path(out)
    result_path = out / "round12a-model-raw.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    fixture_path = Path(fixture_path)
    fixture = json.loads(fixture_path.read_text())
    workloads, queue_sha = _load_queue(queue_path, version=version)
    schedule = build_schedule(
        fixture,
        version=version,
        mode=mode,
        block=block,
        workloads=workloads,
    )
    report = {
        "schema": 1,
        "status": "RUNNING",
        "stage": "P2_FIRST_NONKT_REQUEST" if version == 173 else "P3_KT_NONKT_ABBA",
        "mode": mode,
        "block": block,
        "version": version,
        "profiler": "OFF",
        "timing_primary": "CLIENT_UNPROFILED_E2E",
        "fixture_sha256": hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
        "queue_sha256": queue_sha,
        "workloads": workloads,
        "generation_budget": len(schedule),
        "generation_attempts": 0,
        "abort_controls": 0,
        "automatic_retries": 0,
        "results": [],
    }
    try:
        if mode == "NON_KT":
            runtime = json.loads((out / "nonkt-runtime.json").read_text())
            binding = json.loads((out / "nonkt-offloader-binding.json").read_text())
            if (
                runtime.get("status") != "PASS_ROUND12A_NONKT_SERVER_PREFLIGHT"
                or binding.get("status") != "PASS_ROUND12A_NONKT_OFFLOADER_BINDING"
                or binding.get("decoder_layers") != 48
                or binding.get("cpu_parameter_bytes")
                != binding.get("offloader_accounted_bytes")
            ):
                raise RuntimeError("Round12a non-KT runtime/residency receipt failed")
            report["nonkt_runtime"] = runtime
            report["nonkt_binding"] = binding
        for row in schedule:
            report["generation_attempts"] += 1
            started = time.monotonic_ns()
            with post_fn("/generate", row["request"], 1800) as response:
                stream = consume_stream(response, row["decode_tokens"])
            ended = time.monotonic_ns()
            metrics = timing_metrics(
                started, ended, stream["token_arrival_monotonic_ns"]
            )
            validate_request_wall(metrics["e2e_seconds"], version=version)
            if stream["output_ids"][:2] != [3897, 248046]:
                raise RuntimeError("Round12a request changed the frozen ACK prefix")
            expected_output = (
                SMOKE_EXPECTED_OUTPUT_IDS if version == 173 else W1_EXPECTED_OUTPUT_IDS
            )
            if row["workload"] == "W1" and stream["output_ids"] != expected_output:
                raise RuntimeError("Round12a W1 output differs from the frozen reference")
            if stream.get("finite_logprobs") is not True:
                raise RuntimeError("Round12a request produced non-finite logprobs")
            rid = row["request"]["rid"]
            qsa = validate_candidate_qsa(
                rid, _read_jsonl(out / "context-reference-events.jsonl")
            )
            ple = validate_ple(_read_jsonl(out / "ple-async-events.jsonl"), rid)
            if mode == "KT":
                path = validate_hybrid_path(
                    rid,
                    read_jsonl(out / "hybrid-path.jsonl"),
                    read_jsonl(out / "layerwise-events.jsonl"),
                )
            else:
                path = {
                    "selected_path": "NON_KT_OFFLOADERV1",
                    "cpu_offload_gib": 233,
                    "full_bf16_layers": True,
                }
            report["results"].append(
                {
                    "index": row["index"],
                    "workload": row["workload"],
                    "case_id": row["case_id"],
                    "phase": row["phase"],
                    "repeat": row["repeat"],
                    "request_id": rid,
                    "input_tokens": row["input_tokens"],
                    "decode_tokens": row["decode_tokens"],
                    **metrics,
                    "quality_pass": True,
                    "output_ids": stream["output_ids"],
                    "finite_logprobs": stream["finite_logprobs"],
                    "finish_reason": stream["finish_reason"],
                    "completion_tokens": stream["completion_tokens"],
                    "qsa_dispatch": qsa,
                    "ple": ple,
                    "path": path,
                }
            )
            _save(result_path, report)
        events = _read_jsonl(out / "context-reference-events.jsonl")
        report["qsa_source_binding"] = validate_qsa_binding(events)
        expected = 1 if version == 173 else 4 * len(workloads)
        if report["generation_attempts"] != expected or len(report["results"]) != expected:
            raise RuntimeError("Round12a schedule is incomplete")
        report["status"] = "PASS_ROUND12A_MODEL_DRIVER_RAW"
        _save(result_path, report)
        return report
    except BaseException as error:
        report.update(
            status="FAIL_ROUND12A_MODEL_DRIVER",
            error=f"{type(error).__name__}: {error}",
        )
        _save(result_path, report)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("KT", "NON_KT"), required=True)
    parser.add_argument("--block", choices=("SMOKE", "A", "B"), required=True)
    parser.add_argument("--version", type=int, choices=tuple(ROLES), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--queue", type=Path)
    args = parser.parse_args()
    result = run(
        args.mode,
        args.block,
        args.version,
        args.out,
        args.fixture,
        args.queue,
    )
    print(result["status"], args.mode, args.block, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
