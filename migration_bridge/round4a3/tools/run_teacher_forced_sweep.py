#!/usr/bin/env python3
"""Capture full matched-history logits and optional Layer17 CPU routes."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import time
import urllib.request
from pathlib import Path

import torch


def post_json(url: str, payload: dict) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.load(response)


def files_below(root: Path | None) -> set[Path]:
    return set() if root is None else set(root.rglob("*.pt"))


def wait_for_new(root: Path | None, before: set[Path], minimum: int = 1) -> list[Path]:
    if root is None:
        return []
    for _ in range(100):
        found = sorted(files_below(root) - before)
        if len(found) >= minimum:
            return found
        time.sleep(0.05)
    raise RuntimeError(f"timed out waiting for dumps below {root}")


def select_logit_dump(paths: list[Path], target_position: int) -> tuple[Path, torch.Tensor]:
    matches = []
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        positions = payload.get("model.forward_batch_info.positions")
        logits = payload.get("logits_processor")
        if positions is not None and logits is not None and int(positions[-1]) == target_position:
            matches.append((path, logits[-1].float().contiguous()))
    if len(matches) != 1:
        raise RuntimeError(f"expected one target logit dump for position {target_position}, found {len(matches)}")
    return matches[0]


def select_route_dump(paths: list[Path], cpu_experts: set[int]) -> dict:
    if not paths:
        return {"cpu_hit_count": 0, "cpu_hit_layers": [], "cpu_hit_experts": []}
    sampling_pass = paths[-1]
    payload = torch.load(sampling_pass, map_location="cpu", weights_only=False)
    topk = payload["topk_ids"][-1].tolist()
    hits = sorted(cpu_experts.intersection(int(value) for value in topk))
    return {
        "cpu_hit_count": len(hits),
        "cpu_hit_layers": [int(payload["layer"])] if hits else [],
        "cpu_hit_experts": hits,
        "route_dump": sampling_pass.name,
    }


def history_output_ids(row: dict) -> list[int]:
    """Read both the legacy replay history and the current determinism schema."""
    if "repetitions" in row:
        return [int(value) for value in row["repetitions"][0]["output_ids"]]
    if "reference64" in row:
        return [int(value) for value in row["reference64"]]
    raise RuntimeError(f"history row for {row.get('prompt_id', '<unknown>')} has no output IDs")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:31000")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--baseline-history", type=Path, required=True)
    parser.add_argument("--tensor-dump-dir", type=Path, required=True)
    parser.add_argument("--kt-dump-dir", type=Path)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--positions", type=int, default=64)
    parser.add_argument("--cpu-experts", default="6,8,25,36")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    corpus = json.loads(args.corpus.read_text())
    history = json.loads(args.baseline_history.read_text())
    history_by_id = {row["prompt_id"]: history_output_ids(row) for row in history["rows"]}
    cpu_experts = {int(value) for value in args.cpu_experts.split(",") if value}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock_file = (args.output_dir / ".sweep.lock").open("w")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError(f"another sweep owns {lock_file.name} in {args.output_dir}") from error
    (args.output_dir / "logits").mkdir(exist_ok=True)
    rows = []
    for prompt in corpus["prompts"]:
        generated = history_by_id[prompt["id"]]
        if len(generated) < args.positions:
            raise RuntimeError(f"insufficient baseline history for {prompt['id']}")
        prompt_logits = []
        positions = []
        for token_index in range(args.positions):
            prefix = prompt["input_ids"] + generated[:token_index]
            tensor_before = files_below(args.tensor_dump_dir)
            route_before = files_below(args.kt_dump_dir)
            response = post_json(
                args.base_url.rstrip("/") + "/generate",
                {
                    "input_ids": prefix,
                    "sampling_params": {"temperature": 0, "max_new_tokens": 1, "ignore_eos": True},
                    "return_logprob": True,
                    "logprob_start_len": 0,
                    "top_logprobs_num": 16,
                },
            )
            tensor_new = wait_for_new(args.tensor_dump_dir, tensor_before, minimum=2)
            dump_path, logits = select_logit_dump(tensor_new, len(prefix))
            response_token = int(response["output_ids"][0])
            captured_top1 = int(torch.argmax(logits))
            response_gap = float(logits.max() - logits[response_token])
            if response_gap > 1e-6:
                raise RuntimeError(
                    f"instrumentation mismatch for {prompt['id']}:{token_index}: "
                    f"captured argmax={captured_top1}, response token={response_token}, "
                    f"response gap={response_gap}"
                )
            route_new = wait_for_new(args.kt_dump_dir, route_before, minimum=2) if args.kt_dump_dir else []
            route = select_route_dump(route_new, cpu_experts)
            top_values, top_ids = torch.topk(logits, k=16)
            prompt_logits.append(logits)
            positions.append(
                {
                    "prompt_id": prompt["id"],
                    "token_index": token_index,
                    "baseline_token": int(generated[token_index]),
                    "mode_top1": response_token,
                    "response_token": response_token,
                    "captured_argmax": captured_top1,
                    "response_logit_gap_from_max": response_gap,
                    "top16_ids": [int(value) for value in top_ids],
                    "top16_logits": [float(value) for value in top_values],
                    "tensor_dump": dump_path.name,
                    **route,
                }
            )
            print(prompt["id"], token_index, positions[-1]["mode_top1"], route["cpu_hit_count"], flush=True)
        logit_path = args.output_dir / "logits" / f"{prompt['id']}.pt"
        stacked = torch.stack(prompt_logits)
        torch.save(stacked, logit_path)
        rows.extend(positions)
    manifest = {
        "schema_version": 1,
        "mode": args.mode,
        "corpus_sha256": corpus["sha256"],
        "baseline_history_sha256": history["sha256"],
        "positions_per_prompt": args.positions,
        "row_count": len(rows),
        "logit_dtype": "torch.float32",
        "vocab_size": int(stacked.shape[-1]),
        "rows": rows,
    }
    canonical = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    manifest["sha256"] = hashlib.sha256(canonical).hexdigest()
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
