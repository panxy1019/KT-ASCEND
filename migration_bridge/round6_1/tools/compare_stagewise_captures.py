#!/usr/bin/env python3
"""Summarize Round 6.1 stagewise tensors and the explicit replay oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def load_payload(path: Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "tensor" not in payload:
        raise RuntimeError(f"not a Round6.1 stage payload: {path}")
    return payload["tensor"].detach().cpu().contiguous()


def tensor_sha(tensor: torch.Tensor) -> str:
    raw = tensor.contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def metrics(left: torch.Tensor, right: torch.Tensor) -> dict:
    result = {
        "left_shape": list(left.shape),
        "right_shape": list(right.shape),
        "left_dtype": str(left.dtype),
        "right_dtype": str(right.dtype),
        "left_sha256": tensor_sha(left),
        "right_sha256": tensor_sha(right),
        "exact": bool(torch.equal(left, right)),
    }
    if left.shape != right.shape:
        result["comparable"] = False
        return result
    lhs = left.float()
    rhs = right.float()
    delta = lhs - rhs
    denom = max(float(torch.linalg.vector_norm(lhs)), 1e-12)
    lhs_flat = lhs.flatten()
    rhs_flat = rhs.flatten()
    lhs_norm = float(torch.linalg.vector_norm(lhs_flat))
    rhs_norm = float(torch.linalg.vector_norm(rhs_flat))
    cosine = (
        float(torch.dot(lhs_flat, rhs_flat) / (lhs_norm * rhs_norm))
        if lhs_norm and rhs_norm
        else (1.0 if lhs_norm == rhs_norm else 0.0)
    )
    result.update(
        {
            "comparable": True,
            "max_abs": float(delta.abs().max()) if delta.numel() else 0.0,
            "mean_abs": float(delta.abs().mean()) if delta.numel() else 0.0,
            "relative_l2": float(torch.linalg.vector_norm(delta) / denom),
            "cosine": cosine,
            "exact_equal_count": int((left == right).sum()),
            "element_count": int(left.numel()),
            "left_nonfinite": int((~torch.isfinite(lhs)).sum()),
            "right_nonfinite": int((~torch.isfinite(rhs)).sum()),
        }
    )
    return result


def stage_map(root: Path) -> dict[str, Path]:
    return {path.name: path for path in sorted((root / "stages").glob("*.pt"))}


def select_stage(root: Path, prefix: str, row_count: int) -> torch.Tensor:
    matches = []
    for path in sorted((root / "stages").glob(prefix + "_call*.pt")):
        tensor = load_payload(path)
        if tensor.ndim > 0 and int(tensor.shape[0]) == row_count:
            matches.append(tensor)
    if len(matches) > 1 and all(torch.equal(matches[0], value) for value in matches[1:]):
        return matches[0]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one {row_count}-row stage matching {prefix} below {root}, "
            f"got {len(matches)} distinct or zero candidates"
        )
    return matches[0]


def find_logits(root: Path) -> torch.Tensor:
    request = json.loads((root / "request.json").read_text())
    target_position = int(request["request_input_length"])
    matches = []
    for path in sorted((root / "tensors").rglob("*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        positions = payload.get("model.forward_batch_info.positions")
        logits = payload.get("logits_processor")
        if positions is not None and logits is not None and int(positions[-1]) == target_position:
            matches.append(logits[-1].float().cpu().contiguous())
    if len(matches) != 1:
        raise RuntimeError(f"expected one final-logit tensor below {root}, got {len(matches)}")
    return matches[0]


def common_comparisons(left_root: Path, right_root: Path) -> dict:
    left = stage_map(left_root)
    right = stage_map(right_root)
    return {
        name: metrics(load_payload(left[name]), load_payload(right[name]))
        for name in sorted(left.keys() & right.keys())
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodefer", type=Path, required=True)
    parser.add_argument("--deferred", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--anchor-token", type=int, default=21)
    parser.add_argument("--candidate-token", type=int, default=16520)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    nodefer_logits = find_logits(args.nodefer)
    deferred_logits = find_logits(args.deferred)
    replay_logits = find_logits(args.replay)
    final = {}
    for name, tensor in (
        ("nodefer", nodefer_logits),
        ("deferred", deferred_logits),
        ("replay", replay_logits),
    ):
        final[name] = {
            "top1": int(tensor.argmax()),
            "anchor_logit": float(tensor[args.anchor_token]),
            "candidate_logit": float(tensor[args.candidate_token]),
            "pairwise_margin": float(
                tensor[args.anchor_token] - tensor[args.candidate_token]
            ),
            "sha256": tensor_sha(tensor),
        }
    request = json.loads((args.deferred / "request.json").read_text())
    target_rows = int(request["request_input_length"])
    aliases = {
        "l7_deferred_partial": (
            "cpu_layer07_deferred_output_after_completion",
            "cpu_layer07_explicit_deferred_output",
        ),
        "l8_incremental_input": (
            "cpu_layer08_incremental_input",
            "cpu_layer08_explicit_pending_input",
        ),
        "l8_incremental_output": (
            "cpu_layer08_cpu_output_after_sync",
            "cpu_layer08_explicit_incremental_output",
        ),
    }
    alias_results = {
        label: metrics(
            select_stage(args.deferred, left_name, target_rows),
            select_stage(args.replay, right_name, target_rows),
        )
        for label, (left_name, right_name) in aliases.items()
    }
    result = {
        "schema_version": 1,
        "nodefer_vs_deferred": common_comparisons(args.nodefer, args.deferred),
        "deferred_vs_explicit_replay": common_comparisons(
            args.deferred, args.replay
        ),
        "deferred_vs_explicit_replay_aliases": alias_results,
        "final_logits": final,
        "final_logits_deferred_vs_replay": metrics(
            deferred_logits, replay_logits
        ),
        "final_logits_nodefer_vs_deferred": metrics(
            nodefer_logits, deferred_logits
        ),
    }
    canonical = json.dumps(
        result, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    result["sha256"] = hashlib.sha256(canonical).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
