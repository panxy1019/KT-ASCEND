"""Run the frozen Round16 W1/W2/W4 benchmark against one S0 HTTP server."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import traceback

from round16_integration.b_v5_perf_probe import build_schedule, summarize_workload


def _save(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _request(row: dict) -> dict:
    return {
        "rid": row["request_id"],
        "input_ids": list(row["input_ids"]),
        "stream": row["kind"] == "performance",
        "sampling_params": {
            "temperature": 0,
            "top_p": 1,
            "sampling_seed": 20260920,
            "max_new_tokens": row["decode_tokens"],
            "ignore_eos": row["kind"] == "performance",
        },
        "return_logprob": True,
        "logprob_start_len": -1,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--block", required=True, choices=("S0-A", "S0-B"))
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result_path = args.out / "round16-s0-perf-result.json"
    started_at = time.time()
    report = {
        "schema": 1,
        "status": "RUNNING",
        "block": args.block,
        "profiler": "OFF",
        "generation_attempts": 0,
        "automatic_retries": 0,
        "results": [],
    }
    try:
        from round10_t1_protocol import validate_response
        from round11_t4_driver import consume_stream, post, validate_hybrid_path
        from round15_b_model_driver import (
            read_jsonl,
            validate_gdn_bindings,
            validate_ple,
            validate_qsa,
        )

        fixture = json.loads(args.fixture.read_text())
        schedule = build_schedule(fixture, args.block)
        for row in schedule:
            report["generation_attempts"] += 1
            request = _request(row)
            begin = time.monotonic_ns()
            if row["kind"] == "performance":
                with post("/generate", request, 1800) as response:
                    stream = consume_stream(response, row["decode_tokens"])
                end = time.monotonic_ns()
                arrivals = stream["token_arrival_monotonic_ns"]
                if stream["output_ids"][:2] != [3897, 248046]:
                    raise RuntimeError(f"{row['request_id']} lost ACK/EOS prefix")
                if stream["completion_tokens"] != row["decode_tokens"]:
                    raise RuntimeError(f"{row['request_id']} output budget changed")
                observed = {
                    "completion_tokens": stream["completion_tokens"],
                    "token_ids": stream["output_ids"],
                    "finite_logprobs": stream["finite_logprobs"],
                    "finish_reason": stream["finish_reason"],
                    "ttft_seconds": (arrivals[0] - begin) / 1e9,
                    "tpot_seconds": (
                        (arrivals[-1] - arrivals[0]) / (len(arrivals) - 1) / 1e9
                    ),
                    "e2e_seconds": (end - begin) / 1e9,
                }
            else:
                with post("/generate", request, 1800) as response:
                    value = json.load(response)
                validated = validate_response({**row, "request": request}, value)
                observed = {
                    "completion_tokens": len(validated["output_ids"]),
                    "token_ids": validated["output_ids"],
                    "finite_logprobs": True,
                    "finish_reason": validated["finish_reason"],
                    "text": value.get("text"),
                    "e2e_seconds": (time.monotonic_ns() - begin) / 1e9,
                }
            request_id = row["request_id"]
            observed["path"] = validate_hybrid_path(
                request_id,
                read_jsonl(args.out / "hybrid-path.jsonl"),
                read_jsonl(args.out / "layerwise-events.jsonl"),
            )
            observed["qsa"] = validate_qsa(
                request_id, read_jsonl(args.out / "context-reference-events.jsonl")
            )
            observed["ple"] = validate_ple(
                request_id, read_jsonl(args.out / "ple-async-events.jsonl")
            )
            item = {key: value for key, value in row.items() if key != "input_ids"}
            item.update(observed)
            report["results"].append(item)
            _save(result_path, report)
            print(
                "ROUND16_S0_PERF_REQUEST_PASS",
                request_id,
                f"e2e={observed['e2e_seconds']:.6f}",
                flush=True,
            )
        report["summaries"] = {
            workload: summarize_workload(
                [row for row in report["results"] if row.get("workload") == workload]
            )
            for workload in ("W1", "W2", "W4")
        }
        report["gdn_binding"] = validate_gdn_bindings(
            args.out / "round14a-gdn-branch.jsonl"
        )
        if report["generation_attempts"] != 10:
            raise RuntimeError("S0 performance schedule did not execute exactly 10 requests")
        report.update(
            status="PASS_ROUND16_S0_REQUEST_PERFORMANCE_BLOCK",
            ended_at=time.time(),
        )
        _save(result_path, report)
        print(report["status"], json.dumps(report["summaries"]), flush=True)
        return 0
    except BaseException as error:
        report.update(
            status="FAIL_ROUND16_S0_REQUEST_PERFORMANCE_BLOCK",
            ended_at=time.time(),
            error_type=type(error).__name__,
            error=str(error),
            traceback=traceback.format_exc(),
        )
        _save(result_path, report)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
