#!/usr/bin/env python3
"""Build the immutable Round13 A placement component bundle."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil


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
        repo / "artifacts/round13/round13_placement_component_probe.py": "round13_placement_component_probe.py",
        repo / "artifacts/round13/round13_placement_runtime.py": "round13_placement_runtime.py",
        repo / "artifacts/round13/PLACEMENT_PROFILES.json": "PLACEMENT_PROFILES.json",
        repo / "migration_bridge/round9/tools/probe_identity.py": "probe_identity.py",
    }
    for source, target in copies.items():
        shutil.copy2(source, args.output / target)
    contract = {
        "schema": 1, "status": "FROZEN_NOT_RUN", "stage": "ROUND13_A_PLACEMENT_COMPONENT",
        "profiles_sha256": sha(args.output / "PLACEMENT_PROFILES.json"),
        "runs": {
            "190": {"strategy": "front_loading", "layer": 0, "capacity": 512, "guard_role": "front-a"},
            "191": {"strategy": "front_loading", "layer": 3, "capacity": 0, "guard_role": "front-b"},
            "192": {"strategy": "frequency", "layer": 2, "capacity": 242, "guard_role": "frequency-a"},
            "193": {"strategy": "random_seed42", "layer": 44, "capacity": 44, "guard_role": "random-a"},
            "194": {"strategy": "front_loading", "layer": 0, "capacity": 512,
                    "guard_role": "front-a", "repair_for": 190},
            "195": {"strategy": "front_loading", "layer": 0, "capacity": 512,
                    "guard_role": "front-a", "repair_for": 194},
        },
        "unchanged": ["model revision", "MXFP4 CPU bank", "BF16/NZ29 NPU experts",
            "C128 NUMA2/3/4/7", "route precision", "deferred off", "Dynamic off"],
        "relative_l2_max": 0.01, "repeat_hashes": 3, "automatic_retries": 0,
    }
    (args.output / "CONTRACT.json").write_text(json.dumps(contract, indent=2) + "\n")
    manifest = {str(path.relative_to(args.output)): sha(path)
        for path in sorted(args.output.rglob("*")) if path.is_file()}
    (args.output / "source-sha256.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("PASS_ROUND13_A_COMPONENT_BUNDLE", sha(args.output / "source-sha256.json"))


if __name__ == "__main__":
    main()
