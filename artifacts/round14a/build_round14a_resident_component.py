#!/usr/bin/env python3
"""Build the immutable Round14a incremental-residency component bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


RUNS = {
    "300": {"strategy": "fplus1", "layer": 0, "capacity": 512,
            "result_key": "fplus1_layer0_q512"},
    "301": {"strategy": "fplus1", "layer": 24, "capacity": 32,
            "result_key": "fplus1_layer24_q32"},
    "302": {"strategy": "uplus42", "layer": 0, "capacity": 42,
            "result_key": "uplus42_layer0_q42"},
    "303": {"strategy": "uplus42", "layer": 24, "capacity": 42,
            "result_key": "uplus42_layer24_q42"},
    "304": {"strategy": "fplus1", "layer": 0, "capacity": 512,
            "result_key": "fplus1_layer0_q512", "repair_for": 300},
    "305": {"strategy": "fplus1", "layer": 0, "capacity": 512,
            "result_key": "fplus1_layer0_q512", "repair_for": 304},
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    repo = Path(__file__).resolve().parents[2]
    args.output.mkdir(parents=True)
    copies = {
        repo / "artifacts/round14a/round14a_resident_component_probe.py":
            "round14a_resident_component_probe.py",
        repo / "artifacts/round14a/round14a_resident_runtime.py":
            "round14a_resident_runtime.py",
        repo / "artifacts/round14a/ROUND14A_RESIDENT_PROFILES.json":
            "ROUND14A_RESIDENT_PROFILES.json",
        repo / "migration_bridge/round9/tools/probe_identity.py": "probe_identity.py",
    }
    for source, target in copies.items():
        shutil.copy2(source, args.output / target)
    contract = {
        "schema": 1,
        "status": "FROZEN_NOT_RUN",
        "stage": "ROUND14A_B_INCREMENTAL_RESIDENCY_COMPONENT",
        "resident_profiles_sha256": sha(args.output / "ROUND14A_RESIDENT_PROFILES.json"),
        "runs": RUNS,
        "unchanged": [
            "model revision", "MXFP4 CPU bank", "BF16/NZ29 NPU experts",
            "C128 NUMA2/3/4/7", "route precision", "deferred off", "Dynamic off",
        ],
        "relative_l2_max": 0.01,
        "steady_samples": 5,
        "automatic_retries": 0,
    }
    (args.output / "CONTRACT.json").write_text(json.dumps(contract, indent=2) + "\n")
    manifest = {
        str(path.relative_to(args.output)): sha(path)
        for path in sorted(args.output.rglob("*")) if path.is_file()
    }
    (args.output / "source-sha256.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("PASS_ROUND14A_B_RESIDENT_COMPONENT_BUNDLE", sha(args.output / "source-sha256.json"))


if __name__ == "__main__":
    main()
