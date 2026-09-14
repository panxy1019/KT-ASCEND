#!/usr/bin/env python3
"""Collect all-NPU expert frequency for a frozen corpus through SGLang HTTP."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import shutil
import time
import urllib.request
from pathlib import Path

import torch


def post_json(url: str, payload: dict | None = None):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload or {}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        body = response.read().decode("utf-8")
    if not body:
        return None
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return body


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as reader:
        for chunk in iter(lambda: reader.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:31000")
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--model-config", required=True, type=Path)
    parser.add_argument("--recorder-dir", default="/tmp", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--max-new-tokens", default=32, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
    config = json.loads(args.model_config.read_text(encoding="utf-8"))
    before = set(glob.glob(str(args.recorder_dir / "expert_distribution_recorder_*.pt")))
    cases = []
    post_json(f"{args.base_url}/start_expert_distribution_record")
    try:
        for prompt in corpus["prompts"]:
            started = time.perf_counter()
            response = post_json(
                f"{args.base_url}/generate",
                {
                    "text": prompt["text"],
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": args.max_new_tokens,
                        "ignore_eos": True,
                    },
                    "return_logprob": True,
                    "logprob_start_len": 0,
                },
            )
            ids = response.get("output_ids", [])
            logprobs = [x[0] for x in response.get("meta_info", {}).get("output_token_logprobs", [])]
            if len(ids) != args.max_new_tokens or len(logprobs) != len(ids):
                raise AssertionError(f"incomplete generation for {prompt['id']}")
            if not all(value is not None and math.isfinite(value) for value in logprobs):
                raise AssertionError(f"non-finite logprob for {prompt['id']}")
            cases.append(
                {
                    "id": prompt["id"],
                    "output_ids": ids,
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
    finally:
        post_json(f"{args.base_url}/stop_expert_distribution_record")
        post_json(f"{args.base_url}/dump_expert_distribution_record")

    candidates = set(glob.glob(str(args.recorder_dir / "expert_distribution_recorder_*.pt"))) - before
    if len(candidates) != 1:
        raise RuntimeError(f"expected one new recorder artifact, found {sorted(candidates)}")
    source = Path(next(iter(candidates)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, args.output)
    data = torch.load(args.output, map_location="cpu", weights_only=True)
    logical_count = data["logical_count"]
    expected_shape = (
        int(config["num_hidden_layers"]),
        int(config["n_routed_experts"]),
    )
    if logical_count.ndim != 3 or tuple(logical_count.shape[1:]) != expected_shape:
        raise AssertionError(
            f"frequency shape {tuple(logical_count.shape)} does not end with {expected_shape}"
        )
    summary = {
        "mode": "all_npu",
        "temperature": 0,
        "ignore_eos": True,
        "max_new_tokens": args.max_new_tokens,
        "corpus_sha256": corpus["sha256"],
        "request_count": len(cases),
        "logical_count_shape": list(logical_count.shape),
        "frequency_matrix_shape": list(logical_count.sum(dim=0).shape),
        "total_routes": int(logical_count.sum().item()),
        "artifact_sha256": sha256_file(args.output),
        "source_recorder_file": str(source),
        "cases": cases,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
