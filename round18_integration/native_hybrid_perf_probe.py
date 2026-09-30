"""Profiler-off W1/W2/W4 benchmark for the Round18 native Hybrid model."""

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

from round18_integration.native_hybrid_full_smoke import (
    EXPECTED_TOKENIZER_FILES,
    _sha256,
    _write,
    extract_finite_selected_logprobs,
    validate_known_answer,
)


ROUND18_WORKLOAD_FIXTURE_SHA256 = (
    "21e4f82531a58881f2fa251fda50ef3f9909c589336164b77fe732a77abdca95"
)


def build_schedule(
    fixture: Mapping[str, Any], block: str, *, measurement_count: int = 3
) -> list[dict[str, Any]]:
    if measurement_count < 1:
        raise ValueError("measurement_count must be positive")
    workloads = {
        "W1": ("edge128", 16),
        "W2": ("edge128", 128),
        "W4": ("edge64", 32),
    }
    rows: list[dict[str, Any]] = []
    for workload, (case_id, decode_tokens) in workloads.items():
        case = fixture["cases"][case_id]
        input_ids = [int(value) for value in case["input_ids"]]
        expected_input_tokens = 128 if workload in ("W1", "W2") else 64
        if len(input_ids) != expected_input_tokens:
            raise RuntimeError(
                f"{case_id} contains {len(input_ids)} tokens instead of "
                f"{expected_input_tokens}"
            )
        phases = [("warmup", 0)] + [
            ("measurement", repeat) for repeat in range(1, measurement_count + 1)
        ]
        for phase, repeat in phases:
            rows.append(
                {
                    "kind": "performance",
                    "workload": workload,
                    "case_id": case_id,
                    "phase": phase,
                    "repeat": repeat,
                    "request_id": (
                        f"round18-{block.lower()}-{workload.lower()}-{phase}-{repeat}"
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
            "request_id": f"round18-{block.lower()}-canary-math391",
            "input_ids": [int(value) for value in canary["input_ids"]],
            "input_tokens": len(canary["input_ids"]),
            "decode_tokens": 32,
            "validator": canary["validator"],
        }
    )
    return rows


def summarize_workload(
    rows: Sequence[Mapping[str, Any]], *, expected_measurements: int
) -> dict[str, object]:
    selected = [row for row in rows if row["phase"] == "measurement"]
    if len(selected) != expected_measurements:
        raise RuntimeError(
            f"each workload requires exactly {expected_measurements} measurement "
            f"requests, observed {len(selected)}"
        )
    first = min(int(row["started_monotonic_ns"]) for row in selected)
    last = max(int(row["ended_monotonic_ns"]) for row in selected)
    window = (last - first) / 1e9
    if window <= 0:
        raise RuntimeError("continuous measurement window is not positive")
    tokens = sum(int(row["completion_tokens"]) for row in selected)
    return {
        "measurement_requests": expected_measurements,
        "continuous_window_seconds": window,
        "output_tokens": tokens,
        "serial_output_tokens_per_second": tokens / window,
        "requests_per_second": expected_measurements / window,
        "median_ttft_seconds": statistics.median(
            float(row["ttft_seconds"]) for row in selected
        ),
        "median_tpot_seconds": statistics.median(
            float(row["tpot_seconds"]) for row in selected
        ),
        "median_e2e_seconds": statistics.median(
            float(row["e2e_seconds"]) for row in selected
        ),
        "min_completion_tokens": min(
            int(row["completion_tokens"]) for row in selected
        ),
        "max_completion_tokens": max(
            int(row["completion_tokens"]) for row in selected
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
    started_ns = time.monotonic_ns()
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
                    f"request {request_id} observed foreign output "
                    f"{request_output.request_id}"
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
    ended_ns = time.monotonic_ns()
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
        "started_monotonic_ns": started_ns,
        "ended_monotonic_ns": ended_ns,
        "ttft_seconds": (arrivals[0] - started_ns) / 1e9,
        "tpot_seconds": (
            (arrivals[-1] - arrivals[0]) / (len(arrivals) - 1) / 1e9
            if len(arrivals) > 1
            else None
        ),
        "e2e_seconds": (ended_ns - started_ns) / 1e9,
    }


def _resolved_graph_mode(llm: Any) -> str:
    try:
        mode = llm.llm_engine.vllm_config.compilation_config.cudagraph_mode
    except AttributeError:
        return "UNKNOWN"
    return getattr(mode, "name", str(mode))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--fixtures", required=True, type=Path)
    parser.add_argument("--block", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--measurement-count", type=int, default=3)
    parser.add_argument(
        "--graph-mode",
        choices=(
            "NONE",
            "PIECEWISE",
            "FULL",
            "FULL_DECODE_ONLY",
            "FULL_AND_PIECEWISE",
        ),
        default="NONE",
    )
    args = parser.parse_args()
    started = time.time()
    try:
        import faulthandler

        faulthandler.register(signal.SIGUSR2, file=sys.stderr, all_threads=True)
        if _sha256(args.fixtures) != ROUND18_WORKLOAD_FIXTURE_SHA256:
            raise RuntimeError("frozen W1/W2/W4 workload fixture SHA-256 mismatch")
        tokenizer_hashes = {
            name: _sha256(args.model / name) for name in EXPECTED_TOKENIZER_FILES
        }
        if tokenizer_hashes != EXPECTED_TOKENIZER_FILES:
            raise RuntimeError("tokenizer identity differs from the frozen fixture")
        fixture = json.loads(args.fixtures.read_text())
        if fixture.get("model_revision") != (
            "de4b8e4d43b917e7706784d8bb445c9af86a3540"
        ):
            raise RuntimeError("performance fixture model revision is not frozen")
        schedule = build_schedule(
            fixture, args.block, measurement_count=args.measurement_count
        )

        from vllm import LLM, SamplingParams
        from vllm.config.compilation import (
            CUDAGraphMode,
            CompilationConfig,
            CompilationMode,
        )

        graph_mode = CUDAGraphMode[args.graph_mode]
        enforce_eager = graph_mode is CUDAGraphMode.NONE
        compilation_config = CompilationConfig(
            mode=(
                CompilationMode.NONE
                if enforce_eager
                else CompilationMode.VLLM_COMPILE
            ),
            cudagraph_mode=graph_mode,
            cudagraph_capture_sizes=[] if enforce_eager else [1],
            max_cudagraph_capture_size=0 if enforce_eager else 1,
            compile_sizes=[] if enforce_eager else [1, 32],
            compile_ranges_endpoints=[32],
        )

        construct_started = time.monotonic()
        llm = LLM(
            model=str(args.model),
            worker_cls=(
                "round18_integration.native_hybrid_live_worker."
                "Round18NativeHybridWorker"
            ),
            tensor_parallel_size=1,
            dtype="bfloat16",
            enforce_eager=enforce_eager,
            trust_remote_code=True,
            max_model_len=256,
            max_num_seqs=1,
            max_num_batched_tokens=32,
            enable_chunked_prefill=True,
            gpu_memory_utilization=0.82,
            safetensors_load_strategy="eager",
            disable_log_stats=True,
            seed=20260929,
            compilation_config=compilation_config,
        )
        construct_seconds = time.monotonic() - construct_started
        resolved_mode = _resolved_graph_mode(llm)

        results: list[dict[str, Any]] = []
        for row in schedule:
            observed = _run_engine_request(
                llm.llm_engine,
                _params(
                    SamplingParams,
                    max_tokens=int(row["decode_tokens"]),
                    ignore_eos=row["kind"] == "performance",
                ),
                str(row["request_id"]),
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
                "ROUND18_NATIVE_HYBRID_PERF_REQUEST_PASS",
                row["request_id"],
                f"e2e={observed['e2e_seconds']:.6f}",
                flush=True,
            )

        summaries = {
            workload: summarize_workload(
                [row for row in results if row.get("workload") == workload],
                expected_measurements=args.measurement_count,
            )
            for workload in ("W1", "W2", "W4")
        }
        payload = {
            "schema": 1,
            "status": "PASS_ROUND18_NATIVE_HYBRID_REQUEST_PERFORMANCE_BLOCK",
            "started_at": started,
            "ended_at": time.time(),
            "block": args.block,
            "profiler": "OFF",
            "construct_seconds": construct_seconds,
            "model_revision": fixture["model_revision"],
            "fixtures_sha256": _sha256(args.fixtures),
            "tokenizer_files_sha256": tokenizer_hashes,
            "worker_cls": (
                "round18_integration.native_hybrid_live_worker."
                "Round18NativeHybridWorker"
            ),
            "enable_native_hybrid": os.environ.get("ROUND18_ENABLE_NATIVE_HYBRID"),
            "binding_receipt_path": os.environ.get("ROUND18_BINDING_RECEIPT_PATH"),
            "requested_graph_mode": args.graph_mode,
            "resolved_graph_mode": resolved_mode,
            "enforce_eager": enforce_eager,
            "measurement_count": args.measurement_count,
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
            "status": "FAIL_ROUND18_NATIVE_HYBRID_REQUEST_PERFORMANCE_BLOCK",
            "started_at": started,
            "ended_at": time.time(),
            "block": args.block,
            "requested_graph_mode": args.graph_mode,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        _write(args.output, payload)
        print(payload["status"], json.dumps(payload, sort_keys=True), flush=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
