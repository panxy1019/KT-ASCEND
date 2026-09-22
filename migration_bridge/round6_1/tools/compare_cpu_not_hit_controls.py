#!/usr/bin/env python3
"""Compare D1 Deferred and nodefer logits for a whole-request CPU-not-hit input."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def select_pass(root: Path, token_id: int) -> tuple[Path, dict]:
    matches = []
    for path in root.glob("**/Pass*.pt"):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        input_ids = payload["model.forward_batch_info.input_ids"].reshape(-1).tolist()
        if input_ids == [token_id]:
            matches.append((path, payload))
    if len(matches) != 1:
        raise RuntimeError(f"expected one pass for token {token_id} in {root}, got {len(matches)}")
    return matches[0]


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deferred-dump", type=Path, required=True)
    parser.add_argument("--nodefer-dump", type=Path, required=True)
    parser.add_argument("--token-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    deferred_path, deferred_payload = select_pass(args.deferred_dump, args.token_id)
    nodefer_path, nodefer_payload = select_pass(args.nodefer_dump, args.token_id)
    deferred_logits = deferred_payload["logits_processor"].detach().cpu()
    nodefer_logits = nodefer_payload["logits_processor"].detach().cpu()
    delta = (deferred_logits.float() - nodefer_logits.float()).abs()
    result = {
        "schema_version": "round6.1-cpu-not-hit-control-v1",
        "input_ids": [args.token_id],
        "deferred_pass": deferred_path.name,
        "nodefer_pass": nodefer_path.name,
        "shape": list(deferred_logits.shape),
        "dtype": str(deferred_logits.dtype),
        "deferred_sha256": tensor_sha256(deferred_logits),
        "nodefer_sha256": tensor_sha256(nodefer_logits),
        "exact": bool(torch.equal(deferred_logits, nodefer_logits)),
        "max_abs": float(delta.max().item()),
        "mean_abs": float(delta.mean().item()),
        "deferred_top1": int(deferred_logits.argmax(dim=-1).item()),
        "nodefer_top1": int(nodefer_logits.argmax(dim=-1).item()),
    }
    result["pass"] = result["exact"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
