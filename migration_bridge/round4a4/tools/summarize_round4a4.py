#!/usr/bin/env python3
"""Build a compact machine-readable Round 4A.4 evidence index."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output_path = args.output.resolve()
    files = {}
    for path in sorted(args.evidence_dir.glob("*.json")):
        # The index cannot hash itself: doing so would make the recorded digest
        # depend on the previous generation of the same file.
        if path.resolve() == output_path:
            continue
        files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    payload = {"schema_version": 1, "evidence_files": files}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["sha256"] = hashlib.sha256(canonical).hexdigest()
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(payload["sha256"])


if __name__ == "__main__":
    main()
