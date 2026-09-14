#!/usr/bin/env python3
"""Independently validate Round 4A placement JSON/PT artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from placement_lib import file_sha256, validate_profile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", required=True, type=Path)
    parser.add_argument("--placement-dir", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.model_config.read_text(encoding="utf-8"))
    results = {}
    for name in ("p0", "p1", "p2", "p3"):
        manifest_path = args.placement_dir / f"placement_{name}.json"
        pt_path = args.placement_dir / f"placement_{name}.pt"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        loaded = torch.load(pt_path, map_location="cpu", weights_only=True)
        cpu_by_layer = {
            int(row["layer"]): [int(expert) for expert in row["cpu_experts"]]
            for row in manifest["layer_placements"]
        }
        checked = validate_profile(name, config, cpu_by_layer, loaded["logical_count"])
        if file_sha256(pt_path) != manifest["pt_sha256"]:
            raise AssertionError(f"{name}: PT SHA256 mismatch")
        if checked["layer_placements"] != manifest["layer_placements"]:
            raise AssertionError(f"{name}: JSON/PT placement disagreement")
        results[name] = "PASS"
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
