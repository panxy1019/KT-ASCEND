#!/usr/bin/env python3
"""Run exactly one enabled six-request A1 controller inside the model container."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from round16.a.fresh_a1_device import driver
from round16.a.fresh_a1_snapshot import execute_controller, prepare_controller_input


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--block-id", required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--ledger-sha256", required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--observer-receipt-path", required=True)
    args = parser.parse_args()
    controller = prepare_controller_input(
        args.snapshot,
        block_id=args.block_id,
        ledger_path=args.ledger,
        expected_ledger_sha256=args.ledger_sha256,
        expected_source_manifest_sha256=args.source_manifest_sha256,
        output_dir=args.output,
        observer_path=args.observer_receipt_path,
        enabled=True,
        required_budget_state="SINGLE_MATCHING_START",
    )
    receipt = execute_controller(controller, driver)
    if receipt.get("generation_attempts") != 6:
        raise RuntimeError("fresh A1 driver did not spend exactly six attempts")
    print(json.dumps({"status": receipt["status"], "block_id": receipt["block_id"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
