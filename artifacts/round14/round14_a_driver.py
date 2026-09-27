#!/usr/bin/env python3
"""Run one Round14 A KT or Native-TP8 unprofiled comparison block."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time


ROLES = {
    235: ("KT", "A"),
    236: ("NATIVE_TP8", "A"),
    237: ("NATIVE_TP8", "B"),
    238: ("KT", "B"),
    239: ("NATIVE_TP8", "A"),
    240: ("NATIVE_TP8", "B"),
    241: ("NATIVE_TP8", "A"),
    242: ("NATIVE_TP8", "B"),
    247: ("NATIVE_TP8", "A"),
    248: ("NATIVE_TP8", "B"),
    251: ("NATIVE_TP8", "A"),
    252: ("NATIVE_TP8", "B"),
}

PERFORMANCE_WORKLOADS = {
    "W2": ("round11", "edge128", 128),
    "W0": ("round10", "edge64", 2),
    "W1": ("round11", "edge128", 16),
    "W4": ("round10", "edge64", 32),
    "W3": ("round11", "edge512", 16),
}

QUALITY_CASES = {
    "A": ("canary_marker", "canary_math391", "canary_json", "canary_quote"),
    "B": ("canary_transform", "canary_marker", "canary_math391", "canary_json"),
}
QSA_LAYERS = list(range(3, 48, 4))
GDN_LAYERS = [layer for layer in range(48) if layer not in QSA_LAYERS]


def _request(rid, ids, tokens, *, stream, ignore_eos):
    return {
        "rid": rid,
        "input_ids": list(ids),
        "stream": stream,
        "sampling_params": {
            "temperature": 0,
            "top_p": 1,
            "sampling_seed": 20260920,
            "max_new_tokens": tokens,
            "ignore_eos": ignore_eos,
        },
        "return_logprob": True,
        "logprob_start_len": -1,
    }


def build_schedule(fixtures, *, version, backend, block):
    if ROLES.get(version) != (backend, block):
        raise ValueError("Round14 A role/version mismatch")
    profile = "H513_EXPERIMENTAL" if backend == "KT" else "NATIVE_TP8_NO_HYBRID_THRESHOLD"
    rows = []
    for workload, (source, case_id, decode_tokens) in PERFORMANCE_WORKLOADS.items():
        ids = fixtures[source]["cases"][case_id]["input_ids"]
        required_length = {"W0": 64, "W1": 128, "W2": 128, "W3": 512, "W4": 64}[workload]
        if len(ids) != required_length or any(type(token) is not int for token in ids):
            raise ValueError(f"Round14 A {workload} fixture identity mismatch")
        for phase, repeat in (("warmup", 0), ("measurement", 1),
                              ("measurement", 2), ("measurement", 3)):
            rid = (
                f"r14-a-v{version}-{backend.lower()}-{block.lower()}-"
                f"{workload.lower()}-{phase}-{repeat}"
            )
            rows.append({
                "kind": "performance",
                "workload": workload,
                "case_id": case_id,
                "phase": phase,
                "repeat": repeat,
                "input_tokens": len(ids),
                "decode_tokens": decode_tokens,
                "server_profile": profile,
                "request": _request(
                    rid, ids, decode_tokens, stream=True, ignore_eos=True
                ),
            })
    for case_id in QUALITY_CASES[block]:
        case = fixtures["round11"]["cases"][case_id]
        ids = case["input_ids"]
        rid = f"r14-a-v{version}-{backend.lower()}-{block.lower()}-{case_id}-quality"
        rows.append({
            "kind": "quality",
            "case_id": case_id,
            "phase": "quality",
            "validator": case["validator"],
            "input_tokens": len(ids),
            "decode_tokens": 32,
            "server_profile": profile,
            "request": _request(rid, ids, 32, stream=False, ignore_eos=False),
        })
    if len(rows) != 24:
        raise RuntimeError("Round14 A schedule must contain exactly 24 requests")
    return rows


def measurement_window(rows, *, started_ns, ended_ns):
    measured = [row for row in rows if row.get("phase") == "measurement"]
    if len(measured) != 3:
        raise ValueError("Round14 A requires exactly three measurements")
    if ended_ns <= started_ns:
        raise ValueError("Round14 A measurement window must be positive")
    seconds = (ended_ns - started_ns) / 1e9
    tokens = sum(int(row["completion_tokens"]) for row in measured)
    return {
        "seconds": seconds,
        "successful_requests": len(measured),
        "actual_output_tokens": tokens,
        "serial_output_tokens_per_second": tokens / seconds,
        "requests_per_second": len(measured) / seconds,
    }


def validate_kt_qsa(request_id, events):
    rows = [
        row for row in events
        if row.get("status") == "CONTEXT_REFERENCE_QSA_DISPATCH"
        and row.get("request_id") == request_id
    ]
    prefill = sorted({
        row.get("layer") for row in rows if row.get("strategy") == "VECTORIZED"
    })
    decode = sorted({
        row.get("layer") for row in rows
        if row.get("strategy") == "DIRECT_BF16_ROW"
    })
    unexpected = sorted({
        row.get("strategy") for row in rows
        if row.get("strategy") not in {"VECTORIZED", "DIRECT_BF16_ROW"}
    })
    if prefill != QSA_LAYERS or decode != QSA_LAYERS or unexpected:
        raise RuntimeError("Round14 A KT QSA path identity mismatch")
    return {
        "prefill_strategy": "VECTORIZED",
        "decode_strategy": "DIRECT_BF16_ROW",
        "prefill_layers": prefill,
        "decode_layers": decode,
        "dispatch_count": len(rows),
    }


def validate_native_rank_evidence(events):
    if set(events) != set(range(8)):
        raise RuntimeError("Round14 A native evidence does not cover all TP ranks")
    physical = {}
    for rank in range(8):
        rows = events[rank]
        layer_rows = [
            row for row in rows if row.get("status") == "NATIVE_TP8_LAYER_PHYSICAL"
        ]
        qsa_rows = [
            row for row in rows if row.get("status") == "NATIVE_TP8_OLD_QSA_HIT"
        ]
        gdn_rows = [
            row for row in rows if row.get("status") == "NATIVE_TP8_GDN_HIT"
        ]
        if sorted(row.get("layer") for row in layer_rows) != list(range(48)):
            raise RuntimeError(f"rank {rank} lacks all native expert layers")
        if any(
            row.get("rank") != rank
            or row.get("w13_device") != "npu"
            or row.get("w2_device") != "npu"
            or row.get("dtype") != "torch.bfloat16"
            or int(row.get("w13_bytes", 0)) <= 0
            or int(row.get("w2_bytes", 0)) <= 0
            for row in layer_rows
        ):
            raise RuntimeError(f"rank {rank} native expert physical identity mismatch")
        if sorted(row.get("layer") for row in qsa_rows) != QSA_LAYERS:
            raise RuntimeError(f"rank {rank} lacks all old-QSA layer hits")
        if any(
            row.get("rank") != rank
            or row.get("query_heads") != 3
            or row.get("kv_heads") != 1
            or int(row.get("query_tokens", 0)) <= 0
            for row in qsa_rows
        ):
            raise RuntimeError(f"rank {rank} old-QSA geometry mismatch")
        if sorted(row.get("layer") for row in gdn_rows) != GDN_LAYERS:
            raise RuntimeError(f"rank {rank} lacks all native GDN layers")
        if any(
            row.get("rank") != rank
            or row.get("key_heads") != 2
            or row.get("value_heads") != 6
            or row.get("z_heads") != 6
            or row.get("conv_state_device") != "npu"
            or row.get("recurrent_state_device") != "npu"
            or not row.get("conv_state_shape")
            or not row.get("recurrent_state_shape")
            for row in gdn_rows
        ):
            raise RuntimeError(f"rank {rank} native GDN geometry mismatch")
        physical[str(rank)] = {
            "layers": 48,
            "expert_bytes": sum(
                int(row["w13_bytes"]) + int(row["w2_bytes"])
                for row in layer_rows
            ),
        }
    return {
        "ranks": list(range(8)),
        "old_qsa_branch": "QUALIFIED_ORIGINAL",
        "old_qsa_layers": QSA_LAYERS,
        "gdn_layers": GDN_LAYERS,
        "physical": physical,
    }


def _read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise RuntimeError(f"unterminated evidence file {path.name}")
    return [json.loads(line) for line in raw.splitlines()]


def read_native_rank_evidence(out):
    out = Path(out)
    return {
        rank: _read_jsonl(out / f"native-tp8-rank{rank}.jsonl")
        for rank in range(8)
    }


def validate_kt_ple(request_id, events):
    if not any(row.get("status") == "PLE_ASYNC_BOUND" for row in events):
        raise RuntimeError("Round14 A KT PLE ASYNC binding missing")
    rows = [
        row for row in events
        if row.get("status") == "PLE_BATCH_DRAINED"
        and row.get("request_id") == request_id
    ]
    if not rows or not any(row.get("async_engaged") is True for row in rows):
        raise RuntimeError("Round14 A KT request did not drain PLE ASYNC")
    return {
        "drained": len(rows),
        "async": sum(row.get("async_engaged") is True for row in rows),
    }


def validate_native_ple(out, request_id):
    out = Path(out)
    result = {}
    for rank in range(8):
        events = _read_jsonl(out / f"ple-async-events-rank{rank}.jsonl")
        if sum(row.get("status") == "PLE_ASYNC_BOUND" for row in events) != 1:
            raise RuntimeError(f"native rank {rank} PLE ASYNC binding missing")
        drained = [
            row for row in events
            if row.get("status") == "PLE_BATCH_DRAINED"
            and row.get("request_id") == request_id
        ]
        if not drained or not any(row.get("async_engaged") is True for row in drained):
            raise RuntimeError(f"native rank {rank} request did not drain PLE ASYNC")
        result[rank] = {
            "drained_batches": len(drained),
            "async_batches": sum(
                row.get("async_engaged") is True for row in drained
            ),
        }
    return result


def stream_metrics(stream, *, started_ns, ended_ns):
    arrivals = stream.get("token_arrival_monotonic_ns")
    if (
        not isinstance(arrivals, list)
        or not arrivals
        or any(type(value) is not int for value in arrivals)
        or any(right <= left for left, right in zip(arrivals, arrivals[1:]))
        or arrivals[0] < started_ns
        or ended_ns < arrivals[-1]
    ):
        raise RuntimeError("Round14 A has an invalid real-token arrival clock")
    return {
        "ttft_seconds": (arrivals[0] - started_ns) / 1e9,
        "e2e_seconds": (ended_ns - started_ns) / 1e9,
        "tpot_seconds": (
            (arrivals[-1] - arrivals[0]) / (len(arrivals) - 1) / 1e9
            if len(arrivals) > 1 else None
        ),
    }


def _save(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def _backend_evidence(out, row, backend):
    out = Path(out)
    rid = row["request"]["rid"]
    if backend == "KT":
        from round11_t4_driver import validate_hybrid_path

        return {
            "backend": "KT_S0_ROUND13_FINAL",
            "path": validate_hybrid_path(
                rid,
                _read_jsonl(out / "hybrid-path.jsonl"),
                _read_jsonl(out / "layerwise-events.jsonl"),
            ),
            "qsa": validate_kt_qsa(
                rid, _read_jsonl(out / "context-reference-events.jsonl")
            ),
            "ple": validate_kt_ple(
                rid, _read_jsonl(out / "ple-async-events.jsonl")
            ),
        }
    ranks = validate_native_rank_evidence(read_native_rank_evidence(out))
    return {
        "backend": "SGLANG_NATIVE_TP8_BF16",
        "path": "NATIVE_TP8_NO_KT_NO_LAYER_OFFLOAD",
        "qsa": {
            "strategy": "QUALIFIED_ORIGINAL",
            "ranks": ranks["ranks"],
            "layers": ranks["old_qsa_layers"],
            "observation_scope": "ONE_SHOT_ALL_RANK_SOURCE_BRANCH_RECEIPT",
        },
        "gdn": {"ranks": ranks["ranks"], "layers": ranks["gdn_layers"]},
        "physical": ranks["physical"],
        "ple": validate_native_ple(out, rid),
    }


def _live_request_executor(row, out, backend):
    from round11_t4_driver import consume_stream, post
    from round10_t1_protocol import validate_response

    started = time.monotonic_ns()
    if row["kind"] == "performance":
        with post("/generate", row["request"], 1800) as response:
            stream = consume_stream(response, row["decode_tokens"])
        ended = time.monotonic_ns()
        return {
            "started_ns": started,
            "ended_ns": ended,
            **stream,
            "backend_evidence": _backend_evidence(out, row, backend),
        }
    with post("/generate", row["request"], 1800) as response:
        value = json.load(response)
    ended = time.monotonic_ns()
    validated = validate_response(row, value)
    return {
        "started_ns": started,
        "ended_ns": ended,
        "quality_pass": True,
        **validated,
        "backend_evidence": _backend_evidence(out, row, backend),
    }


def _validate_live_identity(out, backend):
    out = Path(out)
    if backend != "NATIVE_TP8":
        return
    runtime = json.loads((out / "native-tp8-runtime.json").read_text())
    if (
        runtime.get("status") != "PASS_ROUND12A_NATIVE_TP8_SERVER_PREFLIGHT"
        or runtime.get("tp_size") != 8
        or runtime.get("qsa_mode") != "QUALIFIED_ORIGINAL"
        or runtime.get("cpu_offload_gb") != 0
        or runtime.get("kt_weight_path") is not None
    ):
        raise RuntimeError("Round14 A native runtime identity mismatch")


def run(version, backend, block, out, fixture10_path, fixture11_path, *,
        request_executor=None):
    out = Path(out)
    result_path = out / "round14-a-raw.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    fixtures = {
        "round10": json.loads(Path(fixture10_path).read_text()),
        "round11": json.loads(Path(fixture11_path).read_text()),
    }
    schedule = build_schedule(
        fixtures, version=version, backend=backend, block=block
    )
    if request_executor is None:
        _validate_live_identity(out, backend)
        request_executor = _live_request_executor
    report = {
        "schema": 1,
        "status": "RUNNING",
        "stage": "ROUND14_A_KT_NATIVE_TP8_SYSTEM_COMPARISON",
        "version": version,
        "backend": backend,
        "block": block,
        "profiler": "OFF",
        "timing": {
            "stream_contract": "EXACTLY_ONE_TOKEN_PER_CUMULATIVE_GROWTH_EVENT",
            "ttft": "FIRST_REAL_TOKEN_MINUS_REQUEST_SUBMISSION",
            "tpot": "LAST_MINUS_FIRST_REAL_TOKEN_DIVIDED_BY_N_MINUS_1",
            "serial_throughput": "ACTUAL_OUTPUT_TOKENS_OVER_FULL_THREE_REQUEST_WINDOW",
        },
        "server_profile": (
            "H513_EXPERIMENTAL_ALL_REQUESTS_SELECTION_EQUIVALENT_FOR_W0_W1_W2_W4"
            if backend == "KT" else "NATIVE_TP8_NO_HYBRID_THRESHOLD"
        ),
        "generation_budget": 24,
        "generation_attempts": 0,
        "abort_controls": 0,
        "automatic_retries": 0,
        "results": [],
        "measurement_windows": {},
    }
    try:
        for row in schedule:
            report["generation_attempts"] += 1
            execution = request_executor(row, out, backend)
            started = execution.pop("started_ns")
            ended = execution.pop("ended_ns")
            if not isinstance(started, int) or not isinstance(ended, int) or ended <= started:
                raise RuntimeError("Round14 A request wall clock is invalid")
            item = {
                key: row[key]
                for key in (
                    "kind", "workload", "case_id", "phase", "repeat",
                    "input_tokens", "decode_tokens", "server_profile",
                )
                if key in row
            }
            item["request_id"] = row["request"]["rid"]
            item["request_started_monotonic_ns"] = started
            item["request_ended_monotonic_ns"] = ended
            if row["kind"] == "performance":
                output_ids = execution.get("output_ids")
                expected = row["decode_tokens"]
                if (
                    not isinstance(output_ids, list)
                    or len(output_ids) != expected
                    or output_ids[:2] != [3897, 248046]
                    or execution.get("completion_tokens") != expected
                    or execution.get("finite_logprobs") is not True
                    or execution.get("finish_reason") != "length"
                ):
                    raise RuntimeError("Round14 A performance output contract failed")
                item.update(execution)
                item.update(stream_metrics(execution, started_ns=started, ended_ns=ended))
            else:
                if execution.get("quality_pass") is not True:
                    raise RuntimeError("Round14 A quality contract failed")
                item.update(execution)
                item["e2e_seconds"] = (ended - started) / 1e9
            report["results"].append(item)
            _save(result_path, report)
        for workload in PERFORMANCE_WORKLOADS:
            rows = [
                row for row in report["results"]
                if row.get("workload") == workload
            ]
            measured = [row for row in rows if row.get("phase") == "measurement"]
            report["measurement_windows"][workload] = measurement_window(
                measured,
                started_ns=min(row["request_started_monotonic_ns"] for row in measured),
                ended_ns=max(row["request_ended_monotonic_ns"] for row in measured),
            )
        quality = [row for row in report["results"] if row.get("quality_pass")]
        if (
            report["generation_attempts"] != 24
            or len(report["results"]) != 24
            or len(quality) != 4
            or set(report["measurement_windows"]) != set(PERFORMANCE_WORKLOADS)
        ):
            raise RuntimeError("Round14 A schedule or budget is incomplete")
        report["status"] = "PASS_ROUND14_A_DRIVER_RAW"
        _save(result_path, report)
        return report
    except BaseException as error:
        report.update(status="FAIL_ROUND14_A_DRIVER", error=repr(error))
        _save(result_path, report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--backend", choices=("KT", "NATIVE_TP8"), required=True)
    parser.add_argument("--block", choices=("A", "B"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fixture10", type=Path, required=True)
    parser.add_argument("--fixture11", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.version, args.backend, args.block, args.out,
        args.fixture10, args.fixture11,
    )
    print(result["status"], args.version, args.backend, args.block, flush=True)


if __name__ == "__main__":
    main()
