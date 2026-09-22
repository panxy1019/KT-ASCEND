#!/usr/bin/env python3
"""Collect deterministic per-rank routed-expert counts for Round 6 S6."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch


def post_json(url: str, payload: dict) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=900) as response:
        return json.load(response)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as reader:
        for chunk in iter(lambda: reader.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:31000")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--positions", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=27)
    parser.add_argument("--num-experts", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--first-moe-layer", type=int, default=1)
    args = parser.parse_args()

    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
    rank_counts = torch.zeros(
        (args.num_layers, args.num_experts, args.top_k), dtype=torch.int64
    )
    rows = []
    for prompt in corpus["prompts"]:
        started = time.perf_counter()
        response = post_json(
            args.base_url.rstrip("/") + "/generate",
            {
                "input_ids": prompt["input_ids"],
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": args.positions,
                    "ignore_eos": True,
                },
                "return_logprob": True,
                "logprob_start_len": 0,
                "return_routed_experts": True,
            },
        )
        output_ids = [int(value) for value in response.get("output_ids", [])]
        logprobs = [
            value[0]
            for value in response.get("meta_info", {}).get(
                "output_token_logprobs", []
            )
        ]
        encoded = response.get("meta_info", {}).get("routed_experts")
        if not isinstance(encoded, str):
            raise RuntimeError(f"missing routed_experts for {prompt['id']}")
        flat = np.frombuffer(base64.b64decode(encoded), dtype=np.int32)
        row_width = args.num_layers * args.top_k
        if flat.size % row_width:
            raise RuntimeError(
                f"invalid routed expert size for {prompt['id']}: {flat.size}"
            )
        routed = flat.reshape(-1, args.num_layers, args.top_k)
        if routed.shape[0] < args.positions:
            raise RuntimeError(
                f"insufficient routed positions for {prompt['id']}: {routed.shape[0]}"
            )
        routed = routed[-args.positions :]
        if len(output_ids) != args.positions or len(logprobs) != args.positions:
            raise RuntimeError(f"incomplete output for {prompt['id']}")
        if not all(value is not None and math.isfinite(value) for value in logprobs):
            raise RuntimeError(f"non-finite output logprob for {prompt['id']}")
        moe_ids = routed[:, args.first_moe_layer :, :]
        if np.any(moe_ids < 0) or np.any(moe_ids >= args.num_experts):
            raise RuntimeError(f"invalid logical expert ID for {prompt['id']}")
        for layer_idx in range(args.first_moe_layer, args.num_layers):
            layer = torch.from_numpy(routed[:, layer_idx, :].copy()).long()
            for rank in range(args.top_k):
                rank_counts[layer_idx, :, rank] += torch.bincount(
                    layer[:, rank], minlength=args.num_experts
                )
        rows.append(
            {
                "prompt_id": prompt["id"],
                "captured_positions": int(routed.shape[0]),
                "returned_route_positions": int(flat.size // row_width),
                "output_ids": output_ids,
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        print(prompt["id"], routed.shape, flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "corpus_sha256": corpus["sha256"],
            "rank_counts": rank_counts,
            "rank_semantics": "rank_0_highest_to_rank_5_lowest_returned_topk",
        },
        args.output,
    )
    total_routes = int(rank_counts.sum().item())
    expected_routes = (
        len(rows)
        * args.positions
        * (args.num_layers - args.first_moe_layer)
        * args.top_k
    )
    if total_routes != expected_routes:
        raise RuntimeError(f"route count {total_routes} != {expected_routes}")
    summary = {
        "schema_version": 1,
        "mode": "all_npu_defer0_s6",
        "corpus_sha256": corpus["sha256"],
        "positions_per_prompt": args.positions,
        "request_count": len(rows),
        "rank_counts_shape": list(rank_counts.shape),
        "total_routes": total_routes,
        "artifact_sha256": file_sha256(args.output),
        "rows": rows,
    }
    summary["sha256"] = canonical_sha256(summary)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: summary[key] for key in ("total_routes", "artifact_sha256", "sha256")}, indent=2))


if __name__ == "__main__":
    main()
