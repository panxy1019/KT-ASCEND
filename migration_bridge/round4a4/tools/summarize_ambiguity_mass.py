#!/usr/bin/env python3
"""Extract compact ambiguity-size and probability-mass diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.validation.read_text())
    payload = {
        "schema_version": 1,
        "source_sha256": source["sha256"],
        "corpus_role": source["corpus_role"],
        "positions": source["stable"] + source["ambiguous"],
        "ambiguous": source["ambiguous"],
        "ambiguity_size_p95": source["ambiguity_size_p95"],
        "ambiguity_size_max": source["ambiguity_size_max"],
        "ambiguity_mass_p95": source["ambiguity_mass_p95"],
        "ambiguity_mass_max": source["ambiguity_mass_max"],
        "cardinality_gate": False,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["sha256"] = hashlib.sha256(canonical).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
