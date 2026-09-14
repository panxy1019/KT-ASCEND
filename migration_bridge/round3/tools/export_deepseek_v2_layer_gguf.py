#!/usr/bin/env python3
"""Export one DeepSeek-V2 routed-expert layer to an F16 LLAMAFILE GGUF."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from gguf import GGUFWriter
from safetensors import safe_open


PROJECTIONS = {
    "gate": "gate_proj",
    "up": "up_proj",
    "down": "down_proj",
}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--layer", required=True, type=int)
    parser.add_argument("--selected-expert", required=True, type=int)
    parser.add_argument(
        "--output-dtype", choices=("f32", "f16"), default="f32"
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    output = args.output.resolve()
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    index = json.loads(
        (model_dir / "model.safetensors.index.json").read_text(encoding="utf-8")
    )["weight_map"]
    num_experts = int(config["n_routed_experts"])
    top_k = int(config["num_experts_per_tok"])
    hidden_size = int(config["hidden_size"])
    intermediate_size = int(config["moe_intermediate_size"])
    num_layers = int(config["num_hidden_layers"])
    if not 0 <= args.layer < num_layers:
        raise ValueError(f"layer must be in [0, {num_layers}), got {args.layer}")
    if not 0 <= args.selected_expert < num_experts:
        raise ValueError(
            f"selected-expert must be in [0, {num_experts}), got {args.selected_expert}"
        )

    source_keys: dict[str, list[str]] = {}
    source_hashes: dict[str, dict[str, str]] = {}
    exported: dict[str, np.ndarray] = {}
    for short_name, hf_name in PROJECTIONS.items():
        keys = [
            f"model.layers.{args.layer}.mlp.experts.{expert}.{hf_name}.weight"
            for expert in range(num_experts)
        ]
        missing = [key for key in keys if key not in index]
        if missing:
            raise KeyError(f"missing source tensors: {missing[:3]}")
        shards = {index[key] for key in keys}
        if len(shards) != 1:
            raise ValueError(f"projection {short_name} spans shards: {sorted(shards)}")
        shard_name = next(iter(shards))
        tensors = []
        hashes = {}
        with safe_open(model_dir / shard_name, framework="pt", device="cpu") as reader:
            for key in keys:
                tensor = reader.get_tensor(key).contiguous()
                hashes[key] = _sha256_bytes(_tensor_bytes(tensor))
                tensors.append(tensor)
        output_torch_dtype = (
            torch.float32 if args.output_dtype == "f32" else torch.float16
        )
        stacked = torch.stack(tensors).float().to(output_torch_dtype).contiguous()
        exported[short_name] = stacked.numpy()
        source_keys[short_name] = keys
        source_hashes[short_name] = hashes

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = GGUFWriter(str(output), "llama")
    writer.add_block_count(num_layers)
    writer.add_uint32("llama.expert_count", num_experts)
    writer.add_uint32("llama.expert_used_count", top_k)
    writer.add_uint32("llama.embedding_length", hidden_size)
    writer.add_uint32("llama.expert_feed_forward_length", intermediate_size)
    writer.add_string("round3.source_model_dir", str(model_dir))
    writer.add_uint32("round3.source_layer", args.layer)
    output_numpy_dtype = np.float32 if args.output_dtype == "f32" else np.float16
    for short_name, array in exported.items():
        writer.add_tensor(
            f"blk.{args.layer}.ffn_{short_name}_exps.weight",
            np.asarray(array, dtype=output_numpy_dtype, order="C"),
        )
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    manifest = {
        "model_dir": str(model_dir),
        "layer": args.layer,
        "num_experts": num_experts,
        "top_k": top_k,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "output_dtype": str(output_numpy_dtype),
        "output_sha256": _sha256_bytes(output.read_bytes()),
        "output_size": output.stat().st_size,
        "tool_sha256": _sha256_bytes(Path(__file__).read_bytes()),
        "source_keys": source_keys,
        "source_tensor_sha256": source_hashes,
        "selected_expert": args.selected_expert,
        "selected_source_sha256": {
            name: hashes[keys[args.selected_expert]]
            for name, (hashes, keys) in {
                name: (source_hashes[name], source_keys[name])
                for name in PROJECTIONS
            }.items()
        },
    }
    output.with_suffix(output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
