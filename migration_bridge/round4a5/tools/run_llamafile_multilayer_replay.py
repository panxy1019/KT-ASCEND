#!/usr/bin/env python3
"""Replay captured CPU MoE inputs without SGLang or NPU execution.

The tool loads the frozen P2 GGUF into one LLAMAFILE wrapper per requested
layer, then repeatedly calls those wrappers in layer order using the captured
CPU-boundary tensors.  It is a root-cause diagnostic: a failure here isolates
the CPU backend/worker pool from SGLang and CANN lifecycle effects.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path

import torch

from kt_kernel.utils.llamafile import LlamafileMoEWrapper


P2_CPU_EXPERTS = {
    1: [31, 43, 50, 57],
    9: [38, 41, 45, 46],
    17: [6, 8, 25, 36],
    26: [10, 26, 30, 56],
}


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def load_capture(dump_dir: Path, layer: int, pass_index: int) -> dict:
    pattern = dump_dir / f"layer{layer:02d}-pass{pass_index:05d}-pid*.pt"
    paths = sorted(glob.glob(str(pattern)))
    if len(paths) != 1:
        raise SystemExit(
            f"expected one capture for layer={layer} pass={pass_index}, found {len(paths)}"
        )
    payload = torch.load(paths[0], map_location="cpu", weights_only=False)
    required = ("hidden_states", "topk_ids", "topk_weights")
    missing = [key for key in required if key not in payload]
    if missing:
        raise SystemExit(f"capture {paths[0]} lacks {missing}")
    return payload


def build_wrapper(layer: int, args: argparse.Namespace) -> LlamafileMoEWrapper:
    mask = torch.ones(64, dtype=torch.bool)
    mask[P2_CPU_EXPERTS[layer]] = False
    wrapper = LlamafileMoEWrapper(
        layer_idx=layer,
        num_experts=64,
        num_experts_per_tok=6,
        hidden_size=2048,
        moe_intermediate_size=1408,
        gpu_experts_mask=mask,
        cpuinfer_threads=args.cpuinfer_threads,
        threadpool_count=1,
        weight_path=str(args.gguf),
        chunked_prefill_size=args.chunked_prefill_size,
        max_deferred_experts_per_token=0,
        method="LLAMAFILE",
        numa_nodes=[0],
    )
    wrapper.load_weights(torch.arange(64, dtype=torch.int32))
    return wrapper


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--dump-dir", type=Path, required=True)
    parser.add_argument("--layers", default="1,17")
    parser.add_argument("--pass-index", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--cpuinfer-threads", type=int, required=True)
    parser.add_argument("--chunked-prefill-size", type=int, default=512)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    layers = [int(value.strip()) for value in args.layers.split(",") if value.strip()]
    if not layers or any(layer not in P2_CPU_EXPERTS for layer in layers):
        raise SystemExit(f"layers must be drawn from {sorted(P2_CPU_EXPERTS)}")
    if args.repeats <= 0 or args.cpuinfer_threads <= 0:
        raise SystemExit("--repeats and --cpuinfer-threads must be positive")

    captures = {layer: load_capture(args.dump_dir, layer, args.pass_index) for layer in layers}
    LlamafileMoEWrapper._gguf_loader_instance = None
    wrappers = {layer: build_wrapper(layer, args) for layer in layers}

    rows = []
    for index in range(args.repeats):
        layer_rows = []
        for layer in layers:
            payload = captures[layer]
            output = wrappers[layer].forward(
                payload["hidden_states"].to(dtype=torch.bfloat16),
                payload["topk_ids"].to(dtype=torch.int64),
                payload["topk_weights"].to(dtype=torch.float32),
            )
            layer_rows.append(
                {
                    "layer": layer,
                    "input_sha256": tensor_sha256(payload["hidden_states"]),
                    "ids_sha256": tensor_sha256(payload["topk_ids"]),
                    "weights_sha256": tensor_sha256(payload["topk_weights"]),
                    "output_sha256": tensor_sha256(output),
                    "finite": bool(torch.isfinite(output).all().item()),
                }
            )
        rows.append({"repeat": index, "layers": layer_rows})

    unique_by_layer = {
        str(layer): len({row["layers"][position]["output_sha256"] for row in rows})
        for position, layer in enumerate(layers)
    }
    payload = {
        "schema_version": 1,
        "layers": layers,
        "pass_index": args.pass_index,
        "repeats": args.repeats,
        "cpuinfer_threads": args.cpuinfer_threads,
        "all_finite": all(item["finite"] for row in rows for item in row["layers"]),
        "unique_output_hashes_by_layer": unique_by_layer,
        "all_exact": all(count == 1 for count in unique_by_layer.values()),
        "rows": rows,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["sha256"] = hashlib.sha256(canonical).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
