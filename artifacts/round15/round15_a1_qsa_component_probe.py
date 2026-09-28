#!/usr/bin/env python3
"""Real-weight Round15 QSA index-selection screen on physical12."""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

import torch
from safetensors import safe_open


REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
LAYERS = (3, 23, 47)
CONTEXTS = (32, 64, 128, 256)
RATIO = 4
TOPK_BLOCKS = 512
WIDTH = TOPK_BLOCKS * RATIO + RATIO - 1
ITERATIONS = 7
RELATIVE_L2_GATE = 0.001


def relative_l2(actual, expected):
    return float(
        torch.linalg.vector_norm(actual.double() - expected.double())
        / torch.linalg.vector_norm(expected.double()).clamp_min(1e-30)
    )


def tensor_hash(value):
    cpu = value.detach().contiguous().cpu()
    return hashlib.sha256(cpu.view(torch.uint8).numpy().tobytes()).hexdigest()


def norm(value, weight):
    fp32 = value.float()
    return (
        fp32
        * torch.rsqrt(fp32.square().mean(-1, keepdim=True) + 1e-6)
        * (1 + weight.float())
    ).to(value.dtype)


def full_set_device(logical_positions):
    logical = logical_positions.long()
    ids = torch.arange(WIDTH, device=logical.device, dtype=torch.long)
    return torch.where(ids[None, :] <= logical[:, None], ids[None, :], -1)


def selected_set(value):
    return sorted(int(item) for item in value.detach().cpu().reshape(-1) if int(item) >= 0)


def measure(function):
    torch.npu.synchronize()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    began = time.perf_counter_ns()
    start.record()
    output = function()
    end.record()
    torch.npu.synchronize()
    return output, (time.perf_counter_ns() - began) / 1e6, float(start.elapsed_time(end))


def paired_timings(baseline, candidate):
    for function in (baseline, candidate, candidate, baseline):
        measure(function)
    host = {"baseline": [], "candidate": []}
    device = {"baseline": [], "candidate": []}
    functions = {"baseline": baseline, "candidate": candidate}
    pattern = ("baseline", "candidate", "candidate", "baseline")
    step = 0
    while any(len(values) < ITERATIONS for values in host.values()):
        name = pattern[step % len(pattern)]
        step += 1
        if len(host[name]) >= ITERATIONS:
            continue
        _output, host_ms, device_ms = measure(functions[name])
        host[name].append(host_ms)
        device[name].append(device_ms)
    return host, device


def memory_point():
    gc.collect()
    torch.npu.synchronize()
    return {
        "allocated": int(torch.npu.memory_allocated()),
        "reserved": int(torch.npu.memory_reserved()),
    }


def lightning_groups(torch_npu, query, keys, sparse_count):
    """Call the frozen torch-npu schema using contiguous BSND tensors."""
    if query.shape[0] != 1 or query.shape[1:] != (4, 128):
        raise ValueError("Lightning screen requires one [4,128] query")
    if keys.ndim != 2 or keys.shape[1] != 128 or not 1 <= sparse_count <= 2048:
        raise ValueError("Lightning screen key/count geometry invalid")
    weights = torch.full(
        (1, 1, 4),
        1.0 / math.sqrt(128),
        dtype=query.dtype,
        device=query.device,
    )
    output = torch_npu.npu_lightning_indexer(
        query=query.reshape(1, 1, 4, 128).contiguous(),
        key=keys.reshape(1, keys.shape[0], 1, 128).contiguous(),
        weights=weights,
        actual_seq_lengths_query=torch.tensor(
            [1], dtype=torch.int32, device=query.device
        ),
        actual_seq_lengths_key=torch.tensor(
            [keys.shape[0]], dtype=torch.int32, device=query.device
        ),
        layout_query="BSND",
        layout_key="BSND",
        sparse_count=sparse_count,
        sparse_mode=0,
        return_value=False,
    )
    indices = output[0] if isinstance(output, (tuple, list)) else output
    return indices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--identity-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version", type=int, choices=(339, 340), required=True)
    parser.add_argument("--block", choices=("A", "B"), required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if (args.version, args.block) not in ((339, "A"), (340, "B")):
        raise ValueError("Round15 A1 version/block mismatch")

    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root))
    from probe_identity import verify_probe_identity
    from round15_a_qsa_candidate import select_strategy
    from runtime_qsa import compress_index_keys, select_tokens, sparse_gqa_candidate, text_rope
    import torch_npu

    identity = verify_probe_identity(
        model=args.model,
        receipt=args.identity_receipt,
        expected_model=Path("/models") / REVISION,
        expected_receipt_sha256=(
            "7d6cb3db19497303bdd8ceb24efae626e1926a610ab7fbd4406b47d05db4a239"
        ),
        revision=REVISION,
        manifest_sha256=(
            "62e6eca0f2900745dbfd9ba6ed76e60547e0d602acf2d62a9ecac3972028b38e"
        ),
    )
    weight_map = json.loads((args.model / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]

    def load(key):
        with safe_open(str(args.model / weight_map[key]), framework="pt", device="cpu") as handle:
            return handle.get_tensor(key)

    torch.set_num_threads(2)
    torch.npu.set_device(0)
    torch.npu.matmul.allow_hf32 = False
    torch.npu.conv.allow_hf32 = False
    acl = ctypes.CDLL("/usr/local/Ascend/cann-9.0.0/aarch64-linux/lib64/libascendcl.so")
    physical = ctypes.c_int32(-1)
    acl.aclrtGetPhyDevIdByLogicDevId.argtypes = [
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
    ]
    if (
        acl.aclrtGetPhyDevIdByLogicDevId(0, ctypes.byref(physical)) != 0
        or physical.value != 12
        or torch.npu.device_count() != 1
        or list(Path("/dev").glob("davinci[0-9]*")) != [Path("/dev/davinci12")]
    ):
        raise RuntimeError("Round15 A1 component is not isolated to physical12")

    report = {
        "schema": 1,
        "status": "RUNNING",
        "version": args.version,
        "block": args.block,
        "candidate_family": "QSA_INDEX_SELECTION",
        "source_identity": identity,
        "physical_device": 12,
        "layers": list(LAYERS),
        "contexts": list(CONTEXTS),
        "iterations": ITERATIONS,
        "relative_l2_gate": RELATIVE_L2_GATE,
        "timing_gate": "EACH_FULL_INDEX_SELECTION_PARENT_CANDIDATE_OVER_BASELINE_LE_0.85",
        "graph": False,
        "automatic_retries": 0,
        "full_set_rows": [],
        "lightning_rows": [],
        "lightning_disposition": "NOT_RUN",
        "lifecycle": [],
        "allocator": {},
    }
    names = (
        "indexer.index_qk_proj.weight",
        "indexer.q_layernorm.weight",
        "indexer.k_layernorm.weight",
        "q_proj.weight",
        "k_proj.weight",
        "v_proj.weight",
        "q_norm.weight",
        "k_norm.weight",
    )
    try:
        torch.manual_seed(15339 + (1 if args.block == "B" else 0))
        report["allocator"]["initial"] = memory_point()
        lightning_safe = True
        for layer in LAYERS:
            prefix = f"model.language_model.layers.{layer}.self_attn."
            weights = {name: load(prefix + name).to("npu:0") for name in names}
            torch.manual_seed(1533900 + layer + (1000 if args.block == "B" else 0))
            hidden = torch.randn(256, 2560, dtype=torch.bfloat16, device="npu:0")
            projected = hidden @ weights["indexer.index_qk_proj.weight"].T
            positions = torch.arange(256, device="npu:0", dtype=torch.long)
            index_query = text_rope(
                norm(projected[:, :512].reshape(256, 4, 128), weights["indexer.q_layernorm.weight"]),
                positions,
            )
            compressed = compress_index_keys(
                projected[:, 512:],
                weights["indexer.k_layernorm.weight"],
                positions,
                text_rope,
            )
            q_gate = (hidden @ weights["q_proj.weight"].T).reshape(256, 24, 512)
            attention_query = text_rope(
                norm(q_gate[..., :256], weights["q_norm.weight"]), positions
            )
            attention_keys = text_rope(
                norm(
                    (hidden @ weights["k_proj.weight"].T).reshape(256, 2, 256),
                    weights["k_norm.weight"],
                ),
                positions,
            )
            attention_values = (hidden @ weights["v_proj.weight"].T).reshape(256, 2, 256)

            for context in CONTEXTS:
                logical = torch.tensor([context - 1], device="npu:0", dtype=torch.long)
                query = index_query[context - 1 : context]
                keys = compressed[: context // RATIO]
                strategy = select_strategy(
                    query_tokens=1,
                    key_groups=keys.shape[0],
                    topk_blocks=TOPK_BLOCKS,
                    ratio=RATIO,
                    n_heads=query.shape[1],
                    head_dim=query.shape[2],
                    dtype_name=str(query.dtype).removeprefix("torch."),
                )
                if strategy != "FULL_SET_CAUSAL":
                    raise RuntimeError("qualified short Decode did not select full-set strategy")

                def baseline():
                    return select_tokens(
                        query, keys, logical, ratio=RATIO, topk_blocks=TOPK_BLOCKS
                    )

                def candidate():
                    return full_set_device(logical)

                reference = baseline()
                selected = candidate()
                torch.npu.synchronize()
                expected_set = list(range(context))
                if selected_set(reference) != expected_set or selected_set(selected) != expected_set:
                    raise RuntimeError("short-context selected set is not the full causal prefix")
                q = attention_query[context - 1 : context]
                k = attention_keys[:context]
                v = attention_values[:context]
                reference_output = sparse_gqa_candidate(q, k, v, reference)
                candidate_output = sparse_gqa_candidate(q, k, v, selected)
                torch.npu.synchronize()
                output_error = relative_l2(
                    candidate_output.detach().cpu(), reference_output.detach().cpu()
                )
                if not math.isfinite(output_error) or output_error > RELATIVE_L2_GATE:
                    raise RuntimeError(
                        f"full-set downstream QSA numerical gate failed: {output_error}"
                    )
                hashes = []
                for _ in range(3):
                    repeat = sparse_gqa_candidate(q, k, v, candidate())
                    torch.npu.synchronize()
                    hashes.append(tensor_hash(repeat))
                if len(set(hashes)) != 1:
                    raise RuntimeError("full-set downstream QSA repeat hash failed")
                host, device = paired_timings(baseline, candidate)
                base_median = statistics.median(host["baseline"])
                candidate_median = statistics.median(host["candidate"])
                report["full_set_rows"].append(
                    {
                        "layer": layer,
                        "context_tokens": context,
                        "completed_groups": keys.shape[0],
                        "baseline_host_ms": host["baseline"],
                        "candidate_host_ms": host["candidate"],
                        "baseline_device_ms": device["baseline"],
                        "candidate_device_ms": device["candidate"],
                        "baseline_median_ms": base_median,
                        "candidate_median_ms": candidate_median,
                        "candidate_over_baseline": candidate_median / base_median,
                        "downstream_relative_l2": output_error,
                        "repeat_hashes": hashes,
                        "reference_selected_hash": tensor_hash(reference),
                        "candidate_selected_hash": tensor_hash(selected),
                        "selected_set_exact": True,
                    }
                )
                if lightning_safe:
                    try:
                        raw = lightning_groups(torch_npu, query, keys, keys.shape[0])
                        torch.npu.synchronize()
                        groups = selected_set(raw)
                        expected_groups = list(range(keys.shape[0]))
                        safe = groups == expected_groups
                        report["lightning_rows"].append(
                            {
                                "layer": layer,
                                "context_tokens": context,
                                "output_shape": list(raw.shape),
                                "output_dtype": str(raw.dtype),
                                "group_indices": groups,
                                "expected_group_indices": expected_groups,
                                "membership_exact": safe,
                            }
                        )
                        if not safe:
                            lightning_safe = False
                    except BaseException as error:
                        lightning_safe = False
                        report["lightning_rows"].append(
                            {
                                "layer": layer,
                                "context_tokens": context,
                                "status": "INAPPLICABLE_OR_SCHEMA_FAILURE",
                                "error": repr(error),
                            }
                        )
                del reference, selected, reference_output, candidate_output, q, k, v
            del hidden, projected, positions, index_query, compressed
            del attention_query, attention_keys, attention_values, q_gate, weights
            report["allocator"][f"after_layer_{layer}"] = memory_point()

        # Lightning only changes membership once completed groups exceed 512.
        # Exercise that regime separately; it is outside the qualified request
        # range and therefore cannot by itself open the request gate.
        if lightning_safe:
            try:
                torch.manual_seed(1533999 + (1 if args.block == "B" else 0))
                query = torch.randn(1, 4, 128, dtype=torch.bfloat16, device="npu:0")
                keys = torch.randn(520, 128, dtype=torch.bfloat16, device="npu:0")
                logical = torch.tensor([520 * 4 - 1], device="npu:0")
                reference = select_tokens(
                    query, keys, logical, ratio=RATIO, topk_blocks=TOPK_BLOCKS
                )
                raw = lightning_groups(torch_npu, query, keys, TOPK_BLOCKS)
                torch.npu.synchronize()
                reference_groups = sorted({value // RATIO for value in selected_set(reference)})
                lightning = selected_set(raw)
                exact = lightning == reference_groups
                report["lightning_long_context"] = {
                    "groups": 520,
                    "sparse_count": 512,
                    "output_shape": list(raw.shape),
                    "membership_exact": exact,
                    "reference_group_hash": hashlib.sha256(
                        json.dumps(reference_groups).encode()
                    ).hexdigest(),
                    "candidate_group_hash": hashlib.sha256(
                        json.dumps(lightning).encode()
                    ).hexdigest(),
                }
                lightning_safe = exact
                del query, keys, logical, reference, raw
            except BaseException as error:
                lightning_safe = False
                report["lightning_long_context"] = {
                    "status": "INAPPLICABLE_OR_SCHEMA_FAILURE",
                    "error": repr(error),
                }
        report["lightning_disposition"] = (
            "SEMANTICALLY_EXPRESSIBLE_OUTSIDE_CURRENT_REQUEST_RANGE"
            if lightning_safe
            else "REJECTED_OR_INAPPLICABLE_IN_FROZEN_TORCH_NPU_SCHEMA"
        )

        # Twenty same-shape cycles prove that the selected short-context path
        # does not allocate one persistent workspace per request.
        logical = torch.tensor([255], device="npu:0", dtype=torch.long)
        for _ in range(4):
            full_set_device(logical)
        torch.npu.synchronize()
        lifecycle_baseline = memory_point()
        for cycle in range(20):
            selected = full_set_device(logical)
            torch.npu.synchronize()
            point = memory_point()
            report["lifecycle"].append(
                {
                    "cycle": cycle,
                    "allocated": point["allocated"],
                    "reserved": point["reserved"],
                    "hash": tensor_hash(selected),
                }
            )
            del selected
        allocated = [row["allocated"] for row in report["lifecycle"]]
        hashes = {row["hash"] for row in report["lifecycle"]}
        report["lifecycle_verdict"] = {
            "baseline": lifecycle_baseline,
            "allocated_span_bytes": max(allocated) - min(allocated),
            "allocated_growth_bytes": allocated[-1] - allocated[0],
            "repeat_hash_exact": len(hashes) == 1,
            "pass": (
                max(allocated) - min(allocated) <= 65536
                and allocated[-1] - allocated[0] <= 65536
                and len(hashes) == 1
            ),
        }
        if not report["lifecycle_verdict"]["pass"]:
            raise RuntimeError("full-set candidate lifecycle is not bounded")

        ratios = [row["candidate_over_baseline"] for row in report["full_set_rows"]]
        report["full_set_correctness_pass"] = (
            len(report["full_set_rows"]) == len(LAYERS) * len(CONTEXTS)
            and all(
                row["selected_set_exact"]
                and row["downstream_relative_l2"] <= RELATIVE_L2_GATE
                and len(set(row["repeat_hashes"])) == 1
                for row in report["full_set_rows"]
            )
        )
        report["component_gate_pass"] = (
            report["full_set_correctness_pass"]
            and len(ratios) == 12
            and all(ratio <= 0.85 for ratio in ratios)
        )
        report["selected_component_candidate"] = "FULL_SET_CAUSAL_SHORT_CONTEXT"
        report["allocator"]["final"] = memory_point()
        report.update(status="PASS_ROUND15_A1_QSA_COMPONENT_PROCESS")
    except BaseException as error:
        report.update(
            status="FAIL_ROUND15_A1_QSA_COMPONENT_PROCESS",
            error=f"{type(error).__name__}: {error}",
        )
        raise
    finally:
        report["peak_npu_allocated"] = int(torch.npu.max_memory_allocated())
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(report["status"], report.get("component_gate_pass"), flush=True)


if __name__ == "__main__":
    main()
