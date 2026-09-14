#!/usr/bin/env python3
"""Compare defer=0 and defer=1 full logits on the same frozen token history."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import torch


QUANTILES = {"min": 0.0, "p50": 0.5, "p90": 0.9, "p95": 0.95,
             "p99": 0.99, "p99_5": 0.995, "p99_9": 0.999, "max": 1.0}


def canonical_sha256(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def distribution(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    tensor = torch.tensor(values, dtype=torch.float64)
    return {key: float(torch.quantile(tensor, quantile)) for key, quantile in QUANTILES.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--deferred-dir", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--candidate-topk", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    corpus = json.loads(args.corpus.read_text())
    baseline_manifest = json.loads((args.baseline_dir / "manifest.json").read_text())
    deferred_manifest = json.loads((args.deferred_dir / "manifest.json").read_text())
    if not 1 <= args.candidate_topk <= 128:
        raise SystemExit("candidate top-k must be in [1, 128]")
    if baseline_manifest["corpus_sha256"] != corpus["sha256"] or deferred_manifest["corpus_sha256"] != corpus["sha256"]:
        raise SystemExit("corpus SHA mismatch")
    if baseline_manifest["baseline_history_sha256"] != deferred_manifest["baseline_history_sha256"]:
        raise SystemExit("matched-history SHA mismatch")
    if baseline_manifest["positions_per_prompt"] != deferred_manifest["positions_per_prompt"]:
        raise SystemExit("position-count mismatch")
    if deferred_manifest.get("route_telemetry_state") != "POSITION_ALIGNED":
        raise SystemExit("Deferred sweep lacks position-aligned route telemetry")
    baseline_rows = {(row["prompt_id"], row["token_index"]): row for row in baseline_manifest["rows"]}
    deferred_rows = {(row["prompt_id"], row["token_index"]): row for row in deferred_manifest["rows"]}
    if baseline_rows.keys() != deferred_rows.keys() or len(baseline_rows) != baseline_manifest["row_count"]:
        raise SystemExit("position identity mismatch")

    rows: list[dict] = []
    pair_errors: list[float] = []
    strata: dict[str, list[float]] = defaultdict(list)
    for prompt in corpus["prompts"]:
        name = f"{prompt['id']}.pt"
        baseline = torch.load(args.baseline_dir / "logits" / name, map_location="cpu", weights_only=True).float()
        deferred = torch.load(args.deferred_dir / "logits" / name, map_location="cpu", weights_only=True).float()
        if baseline.shape != deferred.shape or baseline.ndim != 2:
            raise SystemExit(f"logit tensor shape mismatch: {name}")
        if baseline.shape[0] != baseline_manifest["positions_per_prompt"]:
            raise SystemExit(f"logit position mismatch: {name}")
        for token_index, (base_logits, deferred_logits) in enumerate(zip(baseline, deferred)):
            key = (prompt["id"], token_index)
            base_row, deferred_row = baseline_rows[key], deferred_rows[key]
            if base_row["baseline_token"] != deferred_row["baseline_token"]:
                raise SystemExit(f"history token mismatch: {key}")
            if not (torch.isfinite(base_logits).all() and torch.isfinite(deferred_logits).all()):
                raise SystemExit(f"non-finite logits: {key}")
            k = min(args.candidate_topk, base_logits.numel())
            base_top = torch.topk(base_logits, k).indices.tolist()
            deferred_top = torch.topk(deferred_logits, k).indices.tolist()
            candidate_ids = sorted(set(base_top + deferred_top))
            anchor = int(torch.argmax(base_logits))
            other_ids = [int(index) for index in candidate_ids if index != anchor]
            base_margins = [float(base_logits[anchor] - base_logits[index]) for index in other_ids]
            deferred_margins = [float(deferred_logits[anchor] - deferred_logits[index]) for index in other_ids]
            pair_distortions = [abs(after - before) for before, after in zip(base_margins, deferred_margins)]
            difference = deferred_logits - base_logits
            base_norm = torch.linalg.vector_norm(base_logits)
            candidate_error = max((float(abs(difference[index])) for index in candidate_ids), default=0.0)
            row = {
                "prompt_id": prompt["id"], "category": prompt["category"], "token_index": token_index,
                "baseline_token": int(base_row["baseline_token"]),
                "baseline_top1_id": anchor, "deferred_top1_id": int(torch.argmax(deferred_logits)),
                "candidate_ids": other_ids, "baseline_pairwise_margins": base_margins,
                "pairwise_abs_distortions": pair_distortions,
                "pairwise_abs_max": max(pair_distortions, default=0.0),
                "full_logit_max_abs": float(torch.max(torch.abs(difference))),
                "full_logit_mean_abs": float(torch.mean(torch.abs(difference))),
                "full_logit_rel_l2": float(torch.linalg.vector_norm(difference) / base_norm.clamp_min(1e-12)),
                "full_logit_cosine": float(torch.nn.functional.cosine_similarity(base_logits, deferred_logits, dim=0)),
                "candidate_logit_max_abs": candidate_error,
                "cpu_hit_count": int(deferred_row["cpu_hit_count"]),
                "cpu_hit_layers": deferred_row["cpu_hit_layers"],
                "cpu_hit_experts": deferred_row["cpu_hit_experts"],
                "actual_deferred_route_count": int(deferred_row["actual_deferred_route_count"]),
                "deferred_producer_layers": deferred_row["deferred_producer_layers"],
                "cpu_route_weights": deferred_row["cpu_route_weights"],
                "route_telemetry_state": "POSITION_ALIGNED",
            }
            rows.append(row)
            pair_errors.extend(pair_distortions)
            strata[f"family:{prompt['category']}"].extend(pair_distortions)
            lower = token_index // 16 * 16
            strata[f"position:{lower}-{lower + 15}"].extend(pair_distortions)
            deferred_count = row["actual_deferred_route_count"]
            strata["deferred:0" if deferred_count == 0 else "deferred:1" if deferred_count == 1 else "deferred:2+"].extend(pair_distortions)
            cpu_hit_count = row["cpu_hit_count"]
            strata["cpu_hit:0" if cpu_hit_count == 0 else "cpu_hit:1" if cpu_hit_count == 1 else "cpu_hit:2+"].extend(pair_distortions)
            for layer_idx in row["deferred_producer_layers"]:
                strata[f"deferred_layer:{layer_idx}"].extend(pair_distortions)
            weight_buckets = set()
            for route_weight in row["cpu_route_weights"]:
                magnitude = abs(route_weight)
                bucket = "0-0.1" if magnitude < 0.1 else "0.1-0.5" if magnitude < 0.5 else "0.5+"
                weight_buckets.add(bucket)
            for bucket in weight_buckets:
                strata[f"route_weight:{bucket}"].extend(pair_distortions)

    payload = {
        "schema_version": 1, "mechanism": "defer0_vs_defer1_matched_history_pairwise_order",
        "corpus_sha256": corpus["sha256"],
        "baseline_history_sha256": baseline_manifest["baseline_history_sha256"],
        "baseline_manifest_sha256": baseline_manifest["sha256"],
        "deferred_manifest_sha256": deferred_manifest["sha256"],
        "candidate_topk": args.candidate_topk, "position_count": len(rows),
        "pair_count": len(pair_errors), "all_finite": True,
        "route_telemetry_state": "POSITION_ALIGNED",
        "pairwise_abs_distortion": distribution(pair_errors),
        "per_position_max_distortion": distribution([row["pairwise_abs_max"] for row in rows]),
        "full_logit_max_abs": distribution([row["full_logit_max_abs"] for row in rows]),
        "full_logit_mean_abs": distribution([row["full_logit_mean_abs"] for row in rows]),
        "full_logit_rel_l2": distribution([row["full_logit_rel_l2"] for row in rows]),
        "candidate_logit_max_abs": distribution([row["candidate_logit_max_abs"] for row in rows]),
        "strata": {key: distribution(values) for key, values in sorted(strata.items())},
        "rows": rows,
    }
    payload["sha256"] = canonical_sha256(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: payload[key] for key in ("position_count", "pair_count", "pairwise_abs_distortion", "full_logit_max_abs", "route_telemetry_state", "sha256")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
