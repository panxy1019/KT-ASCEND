#!/usr/bin/env python3
"""Build the immutable Round12a tied-aware OffloaderV1 component snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys


VERSION = 172
BASE_SHA256 = "98a37555dd1fb6bda68dc5485ca0999dce75a5c43a6df62f86a16f740532442e"
CANDIDATE_SHA256 = "8b68d4012a9bd0fab3bab70ab28db1d7c09b6242af912066d2aedd3f3a3c9006"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(output: Path, inputs: Path) -> Path:
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    round12a = inputs / "artifacts/round12a"
    sys.path.insert(0, str(round12a))
    try:
        from round12a_offloader_patch import patch_offloader_source
    finally:
        sys.path.remove(str(round12a))

    base = round12a / "fixtures/offloader_v1_base.py"
    if sha256(base) != BASE_SHA256:
        raise ValueError("Round12a base OffloaderV1 identity drifted")
    candidate = output / "offloader.py"
    candidate.write_text(patch_offloader_source(base.read_text()))
    if sha256(candidate) != CANDIDATE_SHA256:
        raise ValueError("Round12a candidate OffloaderV1 identity drifted")

    for name in (
        "round12a_tied_offloader_probe.py",
        "round12a_tied_offloader_controller.py",
        "round12a_monitor_owned.py",
    ):
        shutil.copy2(round12a / name, output / name)
    contract = {
        "schema": 1,
        "status": "FROZEN_NOT_RUN",
        "stage": "ROUND12A_TIED_OFFLOADER_ASCEND_GATE",
        "version": VERSION,
        "failure_parent": "ROUND12A_V171_ALLOCATOR_CLEANUP_FAILURE",
        "base_offloader_sha256": BASE_SHA256,
        "candidate_offloader_sha256": CANDIDATE_SHA256,
        "image": "601e76d17e9f",
        "single_change": (
            "deduplicate exact state objects before device copy while retaining "
            "functional_call tie_weights=True"
        ),
        "forbidden_alternative": "tie_weights=False",
        "checks": [
            "tied parameter and tied persistent-buffer aliases",
            "alias identity inside functional_call",
            "bit-exact output against pre-offload reference",
            "unique CPU parameter accounting and immutability",
            "repeat hash and allocator/container cleanup",
        ],
        "automatic_retries": 0,
    }
    (output / "ROUND12A_COMPONENT_CONTRACT.json").write_text(
        json.dumps(contract, indent=2) + "\n"
    )
    manifest = {
        str(path.relative_to(output)): sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "source-sha256.json"
    }
    (output / "source-sha256.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.output, args.inputs)
    print(
        "PASS_ROUND12A_COMPONENT_SNAPSHOT",
        sha256(result / "source-sha256.json"),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
