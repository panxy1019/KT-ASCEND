#!/usr/bin/env python3
"""Derive a Layer-bisection placement from the frozen P2 experts.

This tool never selects new experts.  It keeps P2's four CPU expert IDs for
the requested subset of layers and returns every other P2 layer to NPU only.
The generated ``.pt`` is for root-cause attribution, not P2 acceptance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as reader:
        for chunk in iter(lambda: reader.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p2-placement", required=True, type=Path)
    parser.add_argument("--p2-manifest", required=True, type=Path)
    parser.add_argument("--layers", required=True, help="comma-separated P2 layer IDs")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest-output", required=True, type=Path)
    args = parser.parse_args()

    selected_layers = sorted(
        {int(value.strip()) for value in args.layers.split(",") if value.strip()}
    )
    if not selected_layers:
        raise SystemExit("--layers must not be empty")
    frozen_manifest = json.loads(args.p2_manifest.read_text())
    frozen_cpu_by_layer = {
        int(row["layer"]): [int(expert) for expert in row["cpu_experts"]]
        for row in frozen_manifest["layer_placements"]
    }
    unknown_layers = set(selected_layers) - set(frozen_cpu_by_layer)
    if unknown_layers:
        raise SystemExit(
            "layers are not CPU-enabled in frozen P2: "
            + ", ".join(str(layer) for layer in sorted(unknown_layers))
        )

    source = torch.load(args.p2_placement, map_location="cpu", weights_only=True)
    if "logical_count" not in source:
        raise SystemExit("frozen P2 placement lacks logical_count")
    logical_count = source["logical_count"].clone()
    if logical_count.ndim != 3 or logical_count.shape[0] != 1:
        raise SystemExit(f"unexpected logical_count shape: {tuple(logical_count.shape)}")

    # P2 has zeros only at CPU-owned routes.  Reset all of its four affected
    # layers first, then restore zeros only for the requested diagnostic subset.
    for layer, experts in frozen_cpu_by_layer.items():
        logical_count[0, layer, :] = 1
        if layer in selected_layers:
            logical_count[0, layer, experts] = 0

    output_payload = dict(source)
    output_payload["logical_count"] = logical_count
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_payload, args.output)

    cpu_by_layer = {
        layer: frozen_cpu_by_layer[layer] for layer in selected_layers
    }
    total_moe_experts = int(frozen_manifest["total_moe_experts"])
    selected_cpu_experts = sum(len(experts) for experts in cpu_by_layer.values())
    output_manifest = {
        "schema_version": 1,
        "kind": "round4a5_diagnostic_layer_bisection",
        "source_p2_placement_sha256": file_sha256(args.p2_placement),
        "source_p2_manifest_sha256": file_sha256(args.p2_manifest),
        "selected_layers": selected_layers,
        "cpu_by_layer": cpu_by_layer,
        "selected_cpu_expert_count": selected_cpu_experts,
        "total_moe_experts": total_moe_experts,
        "total_npu_experts": total_moe_experts - selected_cpu_experts,
        "gpu_experts_ratio": (total_moe_experts - selected_cpu_experts)
        / total_moe_experts,
        "placement_sha256": file_sha256(args.output),
        "not_an_acceptance_placement": True,
    }
    canonical = canonical_sha256(output_manifest)
    output_manifest["sha256"] = canonical
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(
        json.dumps(output_manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(output_manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
