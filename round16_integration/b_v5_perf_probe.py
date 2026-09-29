"""Profiler-off W1/W2/W4 request benchmark for the Round16-B eager model."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import statistics
import sys
import time
import traceback
from typing import Any, Mapping, Sequence

from round16_integration.b_v5_smoke_probe import (
    EXPECTED_TOKENIZER_FILES,
    _sha256,
    _write,
    extract_finite_selected_logprobs,
    validate_known_answer,
)


def build_schedule(fixture: Mapping[str, Any], block: str) -> list[dict]:
    workloads = {
        "W1": ("edge128", 16),
        "W2": ("edge128", 128),
        "W4": ("edge64", 32),
    }
    rows: list[dict] = []
    for workload, (case_id, decode_tokens) in workloads.items():
        case = fixture["cases"][case_id]
        input_ids = [int(value) for value in case["input_ids"]]
        for phase, repeat in (
            ("warmup", 0),
            ("measurement", 1),
            ("measurement", 2),
        ):
            rows.append(
                {
                    "kind": "performance",
                    "workload": workload,
                    "case_id": case_id,
                    "phase": phase,
                    "repeat": repeat,
                    "request_id": (
                        f"round16-b-{block.lower()}-{workload.lower()}-{phase}-{repeat}"
                    ),
                    "input_ids": input_ids,
                    "input_tokens": len(input_ids),
                    "decode_tokens": decode_tokens,
                }
            )
    canary = fixture["cases"]["canary_math391"]
    rows.append(
        {
            "kind": "canary",
            "case_id": "canary_math391",
            "phase": "canary",
            "request_id": f"round16-b-{block.lower()}-canary-math391",
            "input_ids": [int(value) for value in canary["input_ids"]],
            "input_tokens": len(canary["input_ids"]),
            "decode_tokens": 32,
            "validator": canary["validator"],
        }
    )
    return rows


def summarize_workload(rows: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    selected = [row for row in rows if row["phase"] == "measurement"]
    if len(selected) != 2:
        raise RuntimeError("each workload requires exactly two measurement requests")
    window = sum(float(row["e2e_seconds"]) for row in selected)
    tokens = sum(int(row["completion_tokens"]) for row in selected)
    return {
        "measurement_requests": 2,
        "window_seconds": window,
        "output_tokens": tokens,
        "serial_output_tokens_per_second": tokens / window,
        "requests_per_second": 2 / window,
        "median_ttft_seconds": statistics.median(
            float(row["ttft_seconds"]) for row in selected
        ),
        "median_tpot_seconds": statistics.median(
            float(row["tpot_seconds"]) for row in selected
        ),
        "median_e2e_seconds": statistics.median(
            float(row["e2e_seconds"]) for row in selected
        ),
    }


def _params(SamplingParams: Any, *, max_tokens: int, ignore_eos: bool) -> Any:
    return SamplingParams(
        temperature=0.0,
        top_p=1.0,
        seed=20260920,
        max_tokens=max_tokens,
        ignore_eos=ignore_eos,
        logprobs=1,
    )


def _run_engine_request(
    engine: Any,
    sampling_params: Any,
    request_id: str,
    input_ids: Sequence[int],
) -> dict[str, object]:
    started = time.monotonic_ns()
    engine.add_request(
        request_id,
        {"prompt_token_ids": list(input_ids)},
        sampling_params,
    )
    arrivals: list[int] = []
    previous_count = 0
    final = None
    steps = 0
    while engine.has_unfinished_requests():
        steps += 1
        if steps > 1024:
            raise RuntimeError(f"request {request_id} exceeded 1024 engine steps")
        for request_output in engine.step():
            if str(request_output.request_id) != request_id:
                raise RuntimeError(
                    f"request {request_id} observed foreign output {request_output.request_id}"
                )
            if len(request_output.outputs) != 1:
                raise RuntimeError("request returned an unexpected completion count")
            completion = request_output.outputs[0]
            current_count = len(completion.token_ids)
            if current_count < previous_count:
                raise RuntimeError("cumulative output token count moved backwards")
            now = time.monotonic_ns()
            arrivals.extend([now] * (current_count - previous_count))
            previous_count = current_count
            final = completion
    ended = time.monotonic_ns()
    if final is None or not arrivals:
        raise RuntimeError(f"request {request_id} produced no visible completion")
    token_ids = [int(value) for value in final.token_ids]
    if len(token_ids) != len(arrivals):
        raise RuntimeError("token arrival count differs from final output count")
    logprobs = extract_finite_selected_logprobs(token_ids, final.logprobs)
    return {
        "request_id": request_id,
        "text": final.text,
        "token_ids": token_ids,
        "selected_logprobs": logprobs,
        "completion_tokens": len(token_ids),
        "finish_reason": final.finish_reason,
        "stop_reason": final.stop_reason,
        "engine_steps": steps,
        "ttft_seconds": (arrivals[0] - started) / 1e9,
        "tpot_seconds": (
            (arrivals[-1] - arrivals[0]) / (len(arrivals) - 1) / 1e9
            if len(arrivals) > 1
            else None
        ),
        "e2e_seconds": (ended - started) / 1e9,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--block", required=True, choices=("B-A", "B-B"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    started = time.time()
    try:
        import faulthandler

        faulthandler.register(signal.SIGUSR2, file=sys.stderr, all_threads=True)
        tokenizer_hashes = {
            name: _sha256(args.model / name) for name in EXPECTED_TOKENIZER_FILES
        }
        if tokenizer_hashes != EXPECTED_TOKENIZER_FILES:
            raise RuntimeError("tokenizer identity differs from the frozen fixture")
        fixture = json.loads(args.fixture.read_text())
        if fixture.get("model_revision") != (
            "de4b8e4d43b917e7706784d8bb445c9af86a3540"
        ):
            raise RuntimeError("performance fixture model revision is not frozen")
        schedule = build_schedule(fixture, args.block)

        from vllm import LLM, SamplingParams

        construct_started = time.monotonic()
        llm = LLM(
            model=str(args.model),
            worker_cls="round16_integration.b2_live_worker.Round16BHybridWorker",
            tensor_parallel_size=1,
            dtype="bfloat16",
            enforce_eager=True,
            trust_remote_code=True,
            max_model_len=256,
            max_num_seqs=1,
            max_num_batched_tokens=32,
            enable_chunked_prefill=True,
            gpu_memory_utilization=0.82,
            safetensors_load_strategy="eager",
            disable_log_stats=True,
            seed=20260929,
        )
        construct_seconds = time.monotonic() - construct_started
        results: list[dict] = []
        for row in schedule:
            observed = _run_engine_request(
                llm.llm_engine,
                _params(
                    SamplingParams,
                    max_tokens=row["decode_tokens"],
                    ignore_eos=row["kind"] == "performance",
                ),
                row["request_id"],
                row["input_ids"],
            )
            if row["kind"] == "performance":
                if observed["completion_tokens"] != row["decode_tokens"]:
                    raise RuntimeError(f"{row['request_id']} output budget changed")
                if observed["token_ids"][:2] != [3897, 248046]:
                    raise RuntimeError(f"{row['request_id']} lost the ACK/EOS prefix")
                if observed["tpot_seconds"] is None:
                    raise RuntimeError(f"{row['request_id']} lacks a TPOT interval")
            else:
                validate_known_answer(str(observed["text"]), row["validator"])
            item = {key: value for key, value in row.items() if key != "input_ids"}
            item.update(observed)
            results.append(item)
            print(
                "ROUND16_B_PERF_REQUEST_PASS",
                row["request_id"],
                f"e2e={observed['e2e_seconds']:.6f}",
                flush=True,
            )

        summaries = {
            workload: summarize_workload(
                [row for row in results if row.get("workload") == workload]
            )
            for workload in ("W1", "W2", "W4")
        }
        payload = {
            "schema": 1,
            "status": "PASS_ROUND16_B_REQUEST_PERFORMANCE_BLOCK",
            "started_at": started,
            "ended_at": time.time(),
            "block": args.block,
            "profiler": "OFF",
            "construct_seconds": construct_seconds,
            "model_revision": fixture["model_revision"],
            "fixture_sha256": _sha256(args.fixture),
            "tokenizer_files_sha256": tokenizer_hashes,
            "worker_cls": "round16_integration.b2_live_worker.Round16BHybridWorker",
            "enable_b2": os.environ.get("ROUND16_B_ENABLE_B2"),
            "generation_attempts": len(schedule),
            "automatic_retries": 0,
            "results": results,
            "summaries": summaries,
        }
        _write(args.output, payload)
        print(payload["status"], json.dumps(summaries, sort_keys=True), flush=True)
        return 0
    except BaseException as error:
        payload = {
            "schema": 1,
            "status": "FAIL_ROUND16_B_REQUEST_PERFORMANCE_BLOCK",
            "started_at": started,
            "ended_at": time.time(),
            "block": args.block,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        _write(args.output, payload)
        print(payload["status"], json.dumps(payload, sort_keys=True), flush=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
