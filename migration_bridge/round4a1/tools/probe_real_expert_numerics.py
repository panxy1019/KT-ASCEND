#!/usr/bin/env python3
"""Compare LLAMAFILE and Ascend BF16 expert stages on captured real inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

from kt_kernel import KTMoEWrapper
from kt_kernel.utils.llamafile import LlamafileMoEWrapper


PROJECTIONS = ("gate", "up", "down")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--gguf", required=True, type=Path)
    parser.add_argument("--capture-dir", required=True, type=Path)
    parser.add_argument(
        "--npu-capture-dir",
        type=Path,
        help="Use production all-NPU stage dumps instead of invoking new GMM shapes",
    )
    parser.add_argument("--layer", required=True, type=int)
    parser.add_argument("--cpu-experts", required=True)
    parser.add_argument("--max-samples-per-expert", default=3, type=int)
    parser.add_argument(
        "--skip-llamafile",
        action="store_true",
        help="Skip heavyweight LLAMAFILE loading and probe R0-R4 only",
    )
    parser.add_argument(
        "--skip-npu-linear",
        action="store_true",
        help="Skip the non-production per-expert NPU F.linear reference",
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as reader:
        for chunk in iter(lambda: reader.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metrics(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    actual = actual.float().cpu()
    reference = reference.float().cpu()
    difference = actual - reference
    norm = float(torch.linalg.vector_norm(reference))
    return {
        "max_abs": float(difference.abs().max()),
        "mean_abs": float(difference.abs().mean()),
        "relative_l2": (
            float(torch.linalg.vector_norm(difference)) / norm if norm else 0.0
        ),
        "cosine": float(
            F.cosine_similarity(actual.flatten(), reference.flatten(), dim=0)
        ),
    }


def load_weights(
    model_dir: Path, weight_map: dict[str, str], layer: int, expert: int
) -> dict[str, torch.Tensor]:
    keys = {
        projection: (
            f"model.layers.{layer}.mlp.experts.{expert}."
            f"{projection}_proj.weight"
        )
        for projection in PROJECTIONS
    }
    by_shard: dict[str, list[tuple[str, str]]] = {}
    for projection, key in keys.items():
        by_shard.setdefault(weight_map[key], []).append((projection, key))
    weights = {}
    for shard, rows in by_shard.items():
        with safe_open(model_dir / shard, framework="pt", device="cpu") as reader:
            for projection, key in rows:
                weights[projection] = reader.get_tensor(key).contiguous()
    return weights


def fp32_stages(
    hidden: torch.Tensor, weights: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    x = hidden.float()
    gate = F.linear(x, weights["gate"].float())
    up = F.linear(x, weights["up"].float())
    activation = F.silu(gate)
    multiplied = activation * up
    down = F.linear(multiplied, weights["down"].float())
    return {
        "gate": gate,
        "up": up,
        "activation": activation,
        "multiplied": multiplied,
        "down": down,
    }


def emulated_bf16_stages(
    hidden: torch.Tensor,
    weights: dict[str, torch.Tensor],
    *,
    round_activation: bool,
    fp32_reference: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    reference = fp32_reference or fp32_stages(hidden, weights)
    gate = reference["gate"].to(torch.bfloat16)
    up = reference["up"].to(torch.bfloat16)
    activation_fp32 = F.silu(gate.float())
    activation = (
        activation_fp32.to(torch.bfloat16)
        if round_activation
        else activation_fp32
    )
    multiplied = (activation.float() * up.float()).to(torch.bfloat16)
    down = F.linear(multiplied.float(), weights["down"].float()).to(
        torch.bfloat16
    )
    return {
        "gate": gate,
        "up": up,
        "activation": activation,
        "multiplied": multiplied,
        "down": down,
    }


def npu_linear_stages(
    hidden: torch.Tensor, weights: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    import torch_npu  # noqa: F401

    x = hidden.to("npu", dtype=torch.bfloat16)
    gate_weight = weights["gate"].to("npu", dtype=torch.bfloat16)
    up_weight = weights["up"].to("npu", dtype=torch.bfloat16)
    down_weight = weights["down"].to("npu", dtype=torch.bfloat16)
    gate = F.linear(x, gate_weight)
    up = F.linear(x, up_weight)
    activation = F.silu(gate)
    multiplied = activation * up
    down = F.linear(multiplied, down_weight)
    torch.npu.synchronize()
    return {
        "gate": gate.cpu(),
        "up": up.cpu(),
        "activation": activation.cpu(),
        "multiplied": multiplied.cpu(),
        "down": down.cpu(),
    }


def npu_gmm_stages(
    hidden: torch.Tensor, weights: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    import torch_npu  # noqa: F401

    x = hidden.to("npu", dtype=torch.bfloat16)
    gate = weights["gate"].to("npu", dtype=torch.bfloat16)
    up = weights["up"].to("npu", dtype=torch.bfloat16)
    down = weights["down"].to("npu", dtype=torch.bfloat16)
    w13 = torch.cat((gate, up), dim=0).transpose(0, 1).contiguous().unsqueeze(0)
    w2 = down.transpose(0, 1).contiguous().unsqueeze(0)
    group_list = torch.tensor([x.shape[0]], dtype=torch.int64, device="npu")
    gate_up = torch.ops.npu.npu_grouped_matmul(
        x=[x],
        weight=[w13],
        bias=None,
        split_item=2,
        group_list_type=1,
        group_type=0,
        group_list=group_list,
        output_dtype=torch.bfloat16,
    )[0]
    intermediate = torch.ops.npu.npu_swiglu(gate_up)
    output = torch.ops.npu.npu_grouped_matmul(
        x=[intermediate],
        weight=[w2],
        bias=None,
        split_item=2,
        group_list_type=1,
        group_type=0,
        group_list=group_list,
        output_dtype=torch.bfloat16,
    )[0]
    torch.npu.synchronize()
    gate_output, up_output = gate_up.chunk(2, dim=-1)
    return {
        "gate": gate_output.cpu(),
        "up": up_output.cpu(),
        "multiplied": intermediate.cpu(),
        "down": output.cpu(),
    }


def captured_npu_stages(
    payload: dict, expert: int, hidden: torch.Tensor
) -> tuple[dict[str, torch.Tensor] | None, dict]:
    counts = payload["expert_tokens"].to(torch.int64)
    start = int(counts[:expert].sum())
    end = start + int(counts[expert])
    routed = payload["routed_input"][start:end]
    matches = torch.where((routed == hidden).all(dim=1))[0]
    metadata = {
        "npu_input_equal": torch.equal(payload["input"], hidden)
        if payload["input"].shape == hidden.shape
        else None,
        "expert_group_start": start,
        "expert_group_end": end,
        "matching_rows": matches.tolist(),
    }
    if matches.numel() == 0:
        return None, metadata
    row = start + int(matches[0])
    gate, up = payload["gmm1_gate_up"][row : row + 1].chunk(2, dim=-1)
    return (
        {
            "gate": gate,
            "up": up,
            "multiplied": payload["swiglu"][row : row + 1],
            "down": payload["gmm2_down"][row : row + 1],
        },
        metadata,
    )


def main() -> None:
    args = parse_args()
    torch.set_num_threads(max(1, len(os.sched_getaffinity(0))))
    print(f"torch_num_threads={torch.get_num_threads()}", flush=True)
    if args.max_samples_per_expert <= 0:
        raise ValueError("--max-samples-per-expert must be positive")
    cpu_experts = [int(item) for item in args.cpu_experts.split(",") if item]
    if not cpu_experts:
        raise ValueError("--cpu-experts must not be empty")
    model_dir = args.model_dir.resolve()
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    weight_map = json.loads(
        (model_dir / "model.safetensors.index.json").read_text(encoding="utf-8")
    )["weight_map"]
    weights_by_expert = {
        expert: load_weights(model_dir, weight_map, args.layer, expert)
        for expert in cpu_experts
    }
    print(f"loaded_weights={sorted(weights_by_expert)}", flush=True)
    source_dtypes = sorted(
        {
            str(tensor.dtype)
            for weights in weights_by_expert.values()
            for tensor in weights.values()
        }
    )

    captures = []
    for path in sorted(args.capture_dir.glob("*.pt")):
        payload = torch.load(path, map_location="cpu")
        if int(payload["layer"]) != args.layer:
            continue
        captures.append((path, payload))
    npu_captures = {}
    if args.npu_capture_dir:
        for path in sorted(args.npu_capture_dir.glob("*.pt")):
            payload = torch.load(path, map_location="cpu")
            if int(payload["layer"]) == args.layer:
                npu_captures[int(payload["pass"])] = payload
        print(f"loaded_npu_passes={sorted(npu_captures)}", flush=True)

    samples: dict[int, list[dict]] = {expert: [] for expert in cpu_experts}
    for path, payload in captures:
        hidden_states = payload["hidden_states"]
        topk_ids = payload["topk_ids"]
        for token_index in range(hidden_states.shape[0]):
            routed = set(int(value) for value in topk_ids[token_index].tolist())
            for expert in routed.intersection(samples):
                if len(samples[expert]) < args.max_samples_per_expert:
                    samples[expert].append(
                        {
                            "capture": path,
                            "pass": int(payload["pass"]),
                            "token_index": token_index,
                            "hidden": hidden_states[token_index : token_index + 1],
                        }
                    )

    wrapper = None
    if not args.skip_llamafile:
        accelerator_mask = torch.ones(
            int(config["n_routed_experts"]), dtype=torch.bool
        )
        accelerator_mask[cpu_experts] = False
        LlamafileMoEWrapper._gguf_loader_instance = None
        LlamafileMoEWrapper._gguf_loader_path = None
        wrapper = KTMoEWrapper(
            layer_idx=args.layer,
            num_experts=int(config["n_routed_experts"]),
            num_experts_per_tok=int(config["num_experts_per_tok"]),
            hidden_size=int(config["hidden_size"]),
            moe_intermediate_size=int(config["moe_intermediate_size"]),
            gpu_experts_mask=accelerator_mask,
            cpuinfer_threads=16,
            threadpool_count=1,
            weight_path=str(args.gguf.resolve()),
            chunked_prefill_size=16,
            max_deferred_experts_per_token=0,
            method="LLAMAFILE",
            numa_nodes=[0],
        )
        wrapper.load_weights(
            torch.arange(int(config["n_routed_experts"]), dtype=torch.int32)
        )

    rows = []
    for expert, expert_samples in samples.items():
        weights = weights_by_expert[expert]
        for sample_index, sample in enumerate(expert_samples):
            hidden = sample.pop("hidden")
            print(f"E{expert} sample={sample_index} stage=start", flush=True)
            ids = torch.full(
                (1, int(config["num_experts_per_tok"])), -1, dtype=torch.int64
            )
            ids[0, 0] = expert
            route_weights = torch.zeros_like(ids, dtype=torch.float32)
            route_weights[0, 0] = 1.0
            llamafile = (
                wrapper.forward(hidden, ids, route_weights).clone()
                if wrapper is not None
                else None
            )
            r0 = fp32_stages(hidden, weights)
            print(f"E{expert} sample={sample_index} stage=r0", flush=True)
            r1 = None if args.skip_npu_linear else npu_linear_stages(hidden, weights)
            print(
                f"E{expert} sample={sample_index} stage=r1"
                f"{'-skipped' if r1 is None else ''}",
                flush=True,
            )
            r2 = emulated_bf16_stages(
                hidden,
                weights,
                round_activation=False,
                fp32_reference=r0,
            )
            r2_every = emulated_bf16_stages(
                hidden,
                weights,
                round_activation=True,
                fp32_reference=r0,
            )
            print(f"E{expert} sample={sample_index} stage=r2", flush=True)
            captured_metadata = None
            if args.npu_capture_dir:
                npu_payload = npu_captures.get(sample["pass"])
                if npu_payload is None:
                    print(
                        f"E{expert} sample={sample_index} stage=r4-missing-pass",
                        flush=True,
                    )
                    continue
                r4, captured_metadata = captured_npu_stages(
                    npu_payload, expert, hidden
                )
                if r4 is None:
                    print(
                        f"E{expert} sample={sample_index} stage=r4-no-hidden-match",
                        flush=True,
                    )
                    continue
            else:
                r4 = npu_gmm_stages(hidden, weights)
            print(f"E{expert} sample={sample_index} stage=r4", flush=True)
            stage_metrics = {}
            for stage in ("gate", "up", "multiplied", "down"):
                stage_metrics[stage] = {
                    "r0_fp32_vs_r4_gmm": metrics(r0[stage], r4[stage]),
                    "r1_linear_vs_r4_gmm": (
                        metrics(r1[stage], r4[stage]) if r1 is not None else None
                    ),
                    "r2_boundaries_vs_r4_gmm": metrics(r2[stage], r4[stage]),
                    "r2_every_boundary_vs_r4_gmm": metrics(
                        r2_every[stage], r4[stage]
                    ),
                }
            row = {
                "expert": expert,
                "sample_index": sample_index,
                "capture_file": sample["capture"].name,
                "pass": sample["pass"],
                "token_index": sample["token_index"],
                "hidden_dtype": str(hidden.dtype),
                "hidden_sha256": hashlib.sha256(
                    hidden.contiguous().view(torch.uint8).numpy().tobytes()
                ).hexdigest(),
                "captured_npu_metadata": captured_metadata,
                "stage_metrics": stage_metrics,
                "final_metrics": {
                    "llamafile_vs_r4_gmm": (
                        metrics(llamafile, r4["down"])
                        if llamafile is not None
                        else None
                    ),
                    "r0_fp32_vs_r4_gmm": metrics(r0["down"], r4["down"]),
                    "r0_bf16_rounded_vs_r4_gmm": metrics(
                        r0["down"].to(torch.bfloat16), r4["down"]
                    ),
                    "r1_linear_vs_r4_gmm": (
                        metrics(r1["down"], r4["down"])
                        if r1 is not None
                        else None
                    ),
                    "r2_boundaries_vs_r4_gmm": metrics(r2["down"], r4["down"]),
                    "r2_every_boundary_vs_r4_gmm": metrics(
                        r2_every["down"], r4["down"]
                    ),
                },
            }
            rows.append(row)
            print(
                f"E{expert} sample={sample_index} "
                f"llamafile/r4="
                f"{row['final_metrics']['llamafile_vs_r4_gmm']['relative_l2'] if llamafile is not None else 'SKIP'} "
                f"r2/r4={row['final_metrics']['r2_boundaries_vs_r4_gmm']['relative_l2']:.6g}",
                flush=True,
            )

    result = {
        "schema_version": 1,
        "layer": args.layer,
        "cpu_experts": cpu_experts,
        "source_weight_dtypes": source_dtypes,
        "gguf": str(args.gguf.resolve()),
        "gguf_sha256": None if args.skip_llamafile else sha256_file(args.gguf),
        "llamafile_skipped": args.skip_llamafile,
        "npu_linear_skipped": args.skip_npu_linear,
        "capture_dir": str(args.capture_dir.resolve()),
        "npu_capture_dir": (
            str(args.npu_capture_dir.resolve()) if args.npu_capture_dir else None
        ),
        "capture_files": [
            {"name": path.name, "sha256": sha256_file(path)}
            for path, _ in captures
        ],
        "sample_counts": {
            str(expert): len(expert_samples)
            for expert, expert_samples in samples.items()
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
