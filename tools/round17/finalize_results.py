#!/usr/bin/env python3
"""Create the frozen Round17 baseline and attribution summaries."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from round17.finalize import aggregate_baseline, baseline_block, load_json, qsa_ceiling


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.artifact_root
    blocks = []
    for name in ("s0-a-v519", "s0-b-v520"):
        raw = root / "raw" / name
        blocks.append(
            baseline_block(
                load_json(raw / "round16-s0-perf-result.json"),
                (raw / "monitor.jsonl").read_text().splitlines(),
            )
        )
    write_json(root / "FRESH_S0_RESULTS.json", aggregate_baseline(blocks))
    profiles = []
    for name in ("profile-a-v512", "profile-b-v514"):
        analysis = load_json(root / "raw" / name / "round17-analysis.json")
        profiles.append(
            {
                "block": name,
                "steady_decode_batches": analysis["steady_decode_batches"],
                "scheduler_wall_ms_per_token": analysis["scheduler_wall_ms_per_token"]["median"],
                "qsa": qsa_ceiling(analysis),
                # The raw analyzer retained its original ``*_ns`` field names,
                # but values were converted to milliseconds before emission.
                "glue_ms_per_token": {
                    key.removesuffix("_ns"): value["median"]
                    for key, value in analysis["glue_ms_per_token"].items()
                },
            }
        )
    write_json(
        root / "QSA_PARENT_ACCOUNT.json",
        {
            "schema": 1,
            "status": "PASS_QSA_FULL_PARENT_ACCOUNT_NO_BOUNDED_CANDIDATE",
            "component_gate_percent": 20,
            "profiles": profiles,
            "decision": "A_NO_BOUNDED_PARENT_CANDIDATE",
        },
    )
    hashes = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "EVIDENCE_SHA256.json":
            hashes.append(
                {
                    "path": str(path.relative_to(root)),
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    write_json(root / "EVIDENCE_SHA256.json", {"schema": 1, "files": hashes})


if __name__ == "__main__":
    main()
