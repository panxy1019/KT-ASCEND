#!/usr/bin/env python3
"""Export selected DeepSeek-V2 MoE layers to one F32 LLAMAFILE GGUF."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from gguf import GGUFWriter
from safetensors import safe_open


PROJECTIONS = {"gate": "gate_proj", "up": "up_proj", "down": "down_proj"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as reader:
        for chunk in iter(lambda: reader.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def parse_layers(value: str) -> list[int]:
    layers = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not layers:
        raise argparse.ArgumentTypeError("--layers must contain at least one layer")
    return layers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--layers", required=True, type=parse_layers)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = args.model_dir.resolve()
    output = args.output.resolve()
    config_path = model_dir / "config.json"
    index_path = model_dir / "model.safetensors.index.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    num_layers = int(config["num_hidden_layers"])
    num_experts = int(config["n_routed_experts"])
    top_k = int(config["num_experts_per_tok"])
    hidden_size = int(config["hidden_size"])
    intermediate_size = int(config["moe_intermediate_size"])
    first_moe = int(config.get("first_k_dense_replace", 0) or 0)
    moe_frequency = int(config.get("moe_layer_freq", 1))
    for layer in args.layers:
        if not 0 <= layer < num_layers:
            raise ValueError(f"layer {layer} is outside [0, {num_layers})")
        if layer < first_moe or layer % moe_frequency:
            raise ValueError(f"layer {layer} is not a routed MoE layer")

    needed_keys = {
        layer: {
            short: [
                f"model.layers.{layer}.mlp.experts.{expert}.{hf_name}.weight"
                for expert in range(num_experts)
            ]
            for short, hf_name in PROJECTIONS.items()
        }
        for layer in args.layers
    }
    missing = [
        key
        for layer_keys in needed_keys.values()
        for projection_keys in layer_keys.values()
        for key in projection_keys
        if key not in weight_map
    ]
    if missing:
        raise KeyError(f"missing source tensors: {missing[:3]}")

    keys_by_shard: dict[str, list[str]] = {}
    for layer_keys in needed_keys.values():
        for projection_keys in layer_keys.values():
            for key in projection_keys:
                keys_by_shard.setdefault(weight_map[key], []).append(key)
    tensors: dict[str, torch.Tensor] = {}
    source_tensor_sha256: dict[str, str] = {}
    for shard_name, keys in sorted(keys_by_shard.items()):
        with safe_open(model_dir / shard_name, framework="pt", device="cpu") as reader:
            for key in keys:
                tensor = reader.get_tensor(key).contiguous()
                source_tensor_sha256[key] = tensor_sha256(tensor)
                tensors[key] = tensor.float().contiguous()

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = GGUFWriter(str(output), "llama")
    writer.add_block_count(num_layers)
    writer.add_uint32("llama.expert_count", num_experts)
    writer.add_uint32("llama.expert_used_count", top_k)
    writer.add_uint32("llama.embedding_length", hidden_size)
    writer.add_uint32("llama.expert_feed_forward_length", intermediate_size)
    writer.add_string("round4a.source_model_dir", str(model_dir))
    writer.add_string("round4a.source_model_revision", args.model_revision)
    writer.add_string("round4a.source_layers", ",".join(map(str, args.layers)))
    exported_shapes = {}
    for layer in args.layers:
        for short_name in PROJECTIONS:
            keys = needed_keys[layer][short_name]
            stacked = torch.stack([tensors.pop(key) for key in keys]).contiguous()
            name = f"blk.{layer}.ffn_{short_name}_exps.weight"
            array = np.asarray(stacked.numpy(), dtype=np.float32, order="C")
            writer.add_tensor(name, array)
            exported_shapes[name] = list(array.shape)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    source_shards = sorted(keys_by_shard)
    manifest = {
        "model_dir": str(model_dir),
        "model_revision": args.model_revision,
        "config_sha256": sha256_file(config_path),
        "safetensors_index_sha256": sha256_file(index_path),
        "source_safetensor_files": source_shards,
        "source_safetensor_sha256": {
            shard: sha256_file(model_dir / shard) for shard in source_shards
        },
        "source_tensor_keys": needed_keys,
        "source_tensor_sha256": source_tensor_sha256,
        "layers": args.layers,
        "num_experts": num_experts,
        "top_k": top_k,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "dtype": "float32",
        "exported_shapes": exported_shapes,
        "exporter_sha256": sha256_file(Path(__file__).resolve()),
        "gguf_sha256": sha256_file(output),
        "gguf_size": output.stat().st_size,
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
