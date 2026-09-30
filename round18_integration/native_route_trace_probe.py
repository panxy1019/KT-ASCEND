"""Fresh W1/W2/W4 native-route capture for Round18 expert-pool economics."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback
from typing import Any, Mapping

from round18_integration.native_hybrid_full_smoke import (
    EXPECTED_TOKENIZER_FILES,
    _sha256,
    _write,
)
from round18_integration.native_hybrid_perf_probe import (
    ROUND18_WORKLOAD_FIXTURE_SHA256,
    _params,
    _run_engine_request,
)


def build_route_trace_schedule(fixture: Mapping[str, Any]) -> list[dict[str, Any]]:
    specs = (
        ("W1", "edge128", 128, 16),
        ("W2", "edge128", 128, 128),
        ("W4", "edge64", 64, 32),
    )
    rows: list[dict[str, Any]] = []
    for workload, case_id, input_tokens, decode_tokens in specs:
        input_ids = [int(value) for value in fixture["cases"][case_id]["input_ids"]]
        if len(input_ids) != input_tokens:
            raise RuntimeError(
                f"{case_id} frozen length is {len(input_ids)}, expected {input_tokens}"
            )
        rows.append(
            {
                "workload": workload,
                "case_id": case_id,
                "request_id": f"round18-route-{workload.lower()}",
                "input_ids": input_ids,
                "input_tokens": input_tokens,
                "decode_tokens": decode_tokens,
                "ignore_eos": True,
            }
        )
    return rows


def _read_events(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"route trace line {line_number} is invalid JSON"
                ) from error
            if row.get("event_index") != len(rows):
                raise RuntimeError("route trace event indices are not contiguous")
            rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--fixtures", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--trace-parent", required=True, type=Path)
    parser.add_argument("--enable-parent", required=True, type=Path)
    args = parser.parse_args()
    started = time.time()
    enable_path = args.enable_parent / "enabled"
    try:
        import faulthandler

        faulthandler.register(signal.SIGUSR2, file=sys.stderr, all_threads=True)
        if _sha256(args.fixtures) != ROUND18_WORKLOAD_FIXTURE_SHA256:
            raise RuntimeError("frozen W1/W2/W4 workload fixture SHA-256 mismatch")
        tokenizer_hashes = {
            name: _sha256(args.model / name) for name in EXPECTED_TOKENIZER_FILES
        }
        if tokenizer_hashes != EXPECTED_TOKENIZER_FILES:
            raise RuntimeError("tokenizer identity differs from frozen fixtures")
        fixture = json.loads(args.fixtures.read_text())
        if fixture.get("model_revision") != (
            "de4b8e4d43b917e7706784d8bb445c9af86a3540"
        ):
            raise RuntimeError("route fixture model revision is not frozen")
        schedule = build_route_trace_schedule(fixture)
        args.trace_parent.mkdir(parents=True, exist_ok=True)
        args.enable_parent.mkdir(parents=True, exist_ok=True)
        trace_path = args.trace_parent / "native-route-trace.jsonl"
        if trace_path.exists():
            raise RuntimeError(f"route trace already exists: {trace_path}")
        if enable_path.exists():
            enable_path.unlink()

        os.environ["ROUND18_ROUTE_TRACE_PARENT"] = str(args.trace_parent)
        os.environ["ROUND18_ROUTE_TRACE_ENABLE_PARENT"] = str(args.enable_parent)

        from vllm import LLM, SamplingParams
        from vllm.config.compilation import (
            CUDAGraphMode,
            CompilationConfig,
            CompilationMode,
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
            compilation_config=CompilationConfig(
                mode=CompilationMode.NONE,
                cudagraph_mode=CUDAGraphMode.NONE,
                cudagraph_capture_sizes=[],
                max_cudagraph_capture_size=0,
                compile_sizes=[],
                compile_ranges_endpoints=[32],
            ),
        )
        construct_seconds = time.monotonic() - construct_started
        if not trace_path.is_file() or trace_path.stat().st_size != 0:
            raise RuntimeError("worker did not create an empty gated route trace")

        enable_path.touch(exist_ok=False)
        request_results: list[dict[str, Any]] = []
        previous_count = 0
        try:
            for row in schedule:
                observed = _run_engine_request(
                    llm.llm_engine,
                    _params(
                        SamplingParams,
                        max_tokens=int(row["decode_tokens"]),
                        ignore_eos=True,
                    ),
                    str(row["request_id"]),
                    row["input_ids"],
                )
                if observed["completion_tokens"] != row["decode_tokens"]:
                    raise RuntimeError(f"{row['request_id']} output budget changed")
                if observed["token_ids"][:2] != [3897, 248046]:
                    raise RuntimeError(f"{row['request_id']} lost the ACK/EOS prefix")
                events = _read_events(trace_path)
                current_count = len(events)
                if current_count <= previous_count:
                    raise RuntimeError(f"{row['request_id']} produced no route events")
                request_events = events[previous_count:current_count]
                layers = sorted({int(event["logical_layer"]) for event in request_events})
                if layers != list(range(48)):
                    raise RuntimeError(
                        f"{row['request_id']} route trace lacks all 48 layers"
                    )
                if any(int(event["top_k"]) != 10 for event in request_events):
                    raise RuntimeError("route trace changed the frozen Top-10 contract")
                request_results.append(
                    {
                        **{key: value for key, value in row.items() if key != "input_ids"},
                        **observed,
                        "route_event_start": previous_count,
                        "route_event_end_exclusive": current_count,
                        "route_event_count": current_count - previous_count,
                        "route_token_rows": sum(
                            int(event["token_rows"]) for event in request_events
                        ),
                    }
                )
                previous_count = current_count
                print(
                    "ROUND18_NATIVE_ROUTE_REQUEST_PASS",
                    row["workload"],
                    f"events={current_count}",
                    flush=True,
                )
        finally:
            if enable_path.exists():
                enable_path.unlink()

        events = _read_events(trace_path)
        payload = {
            "schema": 1,
            "status": "PASS_ROUND18_FRESH_NATIVE_ROUTE_TRACE",
            "started_at": started,
            "ended_at": time.time(),
            "construct_seconds": construct_seconds,
            "profiler": "DIAGNOSTIC_D2H_ROUTE_CAPTURE_NOT_PERFORMANCE",
            "model_revision": fixture["model_revision"],
            "fixtures_sha256": ROUND18_WORKLOAD_FIXTURE_SHA256,
            "tokenizer_files_sha256": tokenizer_hashes,
            "graph_mode": "NONE",
            "logical_experts": 512,
            "top_k": 10,
            "layer_count": 48,
            "route_trace_path": str(trace_path),
            "route_trace_sha256": _sha256(trace_path),
            "route_event_count": len(events),
            "generation_attempts": len(schedule),
            "requests": request_results,
        }
        _write(args.output, payload)
        print(payload["status"], json.dumps(payload, sort_keys=True), flush=True)
        return 0
    except BaseException as error:
        if enable_path.exists():
            enable_path.unlink()
        payload = {
            "schema": 1,
            "status": "FAIL_ROUND18_FRESH_NATIVE_ROUTE_TRACE",
            "started_at": started,
            "ended_at": time.time(),
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        _write(args.output, payload)
        print(payload["status"], json.dumps(payload, sort_keys=True), flush=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
