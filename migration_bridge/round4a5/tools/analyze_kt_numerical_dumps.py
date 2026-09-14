#!/usr/bin/env python3
"""Compare default-off KT numerical dump payloads without altering inference.

The KT EP wrapper writes one ``.pt`` payload per selected layer/pass when
``SGLANG_KT_NUMERICAL_DUMP_DIR`` is enabled.  This tool summarizes tensor
identity and byte hashes for two passes, then reports the first stage that
differs in ascending layer order.  It deliberately never applies a tolerance:
this is a same-path determinism diagnostic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


TENSOR_KEYS = (
    "hidden_states",
    "topk_ids",
    "topk_weights",
    "cpu_output",
    "gpu_output",
    "gpu_routes",
    "merged_output",
)


def tensor_summary(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"expected tensor or None, got {type(value).__name__}")
    contiguous = value.detach().cpu().contiguous()
    raw = contiguous.view(torch.uint8).numpy().tobytes()
    return {
        "shape": list(contiguous.shape),
        "dtype": str(contiguous.dtype),
        "finite": bool(torch.isfinite(contiguous).all().item())
        if contiguous.is_floating_point() or contiguous.is_complex()
        else True,
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def load_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"{path} did not contain a dict payload")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-dir", type=Path, required=True)
    parser.add_argument("--layers", required=True, help="comma-separated layer IDs")
    parser.add_argument("--first-pass", type=int, required=True)
    parser.add_argument("--second-pass", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    layers = [int(value.strip()) for value in args.layers.split(",") if value.strip()]
    if not layers:
        raise SystemExit("--layers must not be empty")

    rows = []
    first_difference = None
    for layer in layers:
        pattern = f"layer{layer:02d}-pass{{pass_index:05d}}-pid*.pt"
        first_paths = sorted(args.dump_dir.glob(pattern.format(pass_index=args.first_pass)))
        second_paths = sorted(args.dump_dir.glob(pattern.format(pass_index=args.second_pass)))
        if len(first_paths) != 1 or len(second_paths) != 1:
            raise SystemExit(
                f"layer {layer}: expected exactly one dump for passes "
                f"{args.first_pass}/{args.second_pass}, found "
                f"{len(first_paths)}/{len(second_paths)}"
            )
        first_payload = load_payload(first_paths[0])
        second_payload = load_payload(second_paths[0])
        stages = {}
        for key in TENSOR_KEYS:
            left = tensor_summary(first_payload.get(key))
            right = tensor_summary(second_payload.get(key))
            equal = left == right
            stages[key] = {"first": left, "second": right, "exact": equal}
            if not equal and first_difference is None:
                first_difference = {"layer": layer, "stage": key}
        rows.append(
            {
                "layer": layer,
                "first_path": first_paths[0].name,
                "second_path": second_paths[0].name,
                "stages": stages,
            }
        )

    payload = {
        "schema_version": 1,
        "dump_dir": str(args.dump_dir),
        "layers": layers,
        "first_pass": args.first_pass,
        "second_pass": args.second_pass,
        "first_difference": first_difference,
        "all_exact": first_difference is None,
        "rows": rows,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["sha256"] = hashlib.sha256(canonical).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
