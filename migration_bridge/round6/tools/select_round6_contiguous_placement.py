#!/usr/bin/env python3
"""Select and freeze nested D1/D2/D3 placements from S6 route ranks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as reader:
        for chunk in iter(lambda: reader.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def rank_experts(rank_counts: torch.Tensor, layer: int) -> list[int]:
    totals = rank_counts[layer].sum(dim=-1)
    lowest = rank_counts[layer, :, -1]
    return sorted(
        range(rank_counts.shape[1]),
        key=lambda expert: (-int(totals[expert]), -int(lowest[expert]), expert),
    )


def layer_selection(rank_counts: torch.Tensor, layer: int) -> dict:
    experts = sorted(rank_experts(rank_counts, layer)[:4])
    total = int(rank_counts[layer, experts, :].sum().item())
    lowest = int(rank_counts[layer, experts, -1].sum().item())
    return {"layer": layer, "cpu_experts": experts, "route_hits": total, "lowest_rank_hits": lowest}


def window_score(rows: list[dict]) -> tuple[int, int, int]:
    return (
        sum(row["route_hits"] for row in rows),
        sum(row["lowest_rank_hits"] for row in rows),
        min(row["route_hits"] for row in rows),
    )


def choose_window(all_rows: dict[int, dict], starts, width: int) -> list[int]:
    candidates = []
    for start in starts:
        layers = list(range(start, start + width))
        rows = [all_rows[layer] for layer in layers]
        candidates.append((window_score(rows), -start, layers))
    return max(candidates)[2]


def build_logical_count(num_layers: int, num_experts: int, selected: list[dict]) -> torch.Tensor:
    logical_count = torch.zeros((1, num_layers, num_experts), dtype=torch.int64)
    logical_count[:, 1:, :] = 1
    for row in selected:
        logical_count[0, row["layer"], row["cpu_experts"]] = 0
    return logical_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-ranks", type=Path, required=True)
    parser.add_argument("--route-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source = torch.load(args.route_ranks, map_location="cpu", weights_only=True)
    summary = json.loads(args.route_summary.read_text(encoding="utf-8"))
    if source["corpus_sha256"] != summary["corpus_sha256"]:
        raise RuntimeError("S6 corpus mismatch between rank artifact and summary")
    if file_sha256(args.route_ranks) != summary["artifact_sha256"]:
        raise RuntimeError("S6 rank artifact SHA mismatch")
    rank_counts = source["rank_counts"].long()
    if tuple(rank_counts.shape) != (27, 64, 6):
        raise RuntimeError(f"unexpected rank count shape {tuple(rank_counts.shape)}")

    all_rows = {layer: layer_selection(rank_counts, layer) for layer in range(1, 27)}
    d3_layers = choose_window(all_rows, range(1, 20), 8)
    d2_layers = choose_window(all_rows, range(d3_layers[0], d3_layers[-1] - 2), 4)
    d1_layers = choose_window(all_rows, range(d2_layers[0], d2_layers[-1]), 2)
    profiles = {"d1": d1_layers, "d2": d2_layers, "d3": d3_layers}
    if not set(d1_layers) <= set(d2_layers) <= set(d3_layers):
        raise RuntimeError("D1/D2/D3 are not nested")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "selection_corpus": "S6",
        "selection_corpus_sha256": summary["corpus_sha256"],
        "route_rank_artifact_sha256": summary["artifact_sha256"],
        "selection_rule": (
            "D3=max total selected-top4 route hits, then rank-6 hits, then minimum layer hits, "
            "then earliest contiguous start; D2 best contiguous four inside D3; "
            "D1 best contiguous two inside D2; per-layer experts use the same ordered criteria"
        ),
        "rank_semantics": source["rank_semantics"],
        "profiles": {},
    }
    for name, layers in profiles.items():
        selected = [all_rows[layer] for layer in layers]
        logical_count = build_logical_count(27, 64, selected)
        pt_path = args.output_dir / f"placement_{name}.pt"
        torch.save({"logical_count": logical_count}, pt_path)
        profile = {
            "layers": layers,
            "selected_layer_count": len(layers),
            "selected_cpu_expert_count": 4 * len(layers),
            "layer_placements": selected,
            "block_terminal_layer": layers[-1],
            "deferred_layers": layers[:-1],
            "terminal_layer_max_deferred": 0,
            "requested_max_deferred": 1,
            "gpu_experts_ratio": (26 * 64 - 4 * len(layers)) / (26 * 64),
            "placement_file": pt_path.name,
            "placement_sha256": file_sha256(pt_path),
            "score": list(window_score(selected)),
        }
        manifest["profiles"][name] = profile
        json_path = args.output_dir / f"placement_{name}.json"
        json_path.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n")
    manifest["sha256"] = canonical_sha256(manifest)
    (args.output_dir / "placement_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
