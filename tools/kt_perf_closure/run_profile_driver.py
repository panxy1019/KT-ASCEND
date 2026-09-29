"""Issue the frozen six-request A1 diagnostic queue without old budget gates."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

from round16.a.fresh_a1_device import driver
from round16.a.fresh_a1_profile import BLOCK_IDS, manifest_payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block", choices=["A", "B"], required=True)
    args = parser.parse_args()
    block_id = BLOCK_IDS[args.block]
    if os.environ.get("ROUND16_A1_BLOCK_ID") != block_id:
        raise RuntimeError("runtime block identity differs from the requested block")
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = []
    started = time.time()
    for item in manifest_payload()["schedule"]:
        request_id = f"{block_id}:{item['ordinal']}:{item['role']}"
        before = time.time()
        result = driver(
            item,
            request_id=request_id,
            output_dir=args.output.parent,
            snapshot_root=args.snapshot,
            environment=os.environ,
        )
        rows.append(
            {
                "ordinal": item["ordinal"],
                "role": item["role"],
                "request_id": request_id,
                "started_at": before,
                "ended_at": time.time(),
                "result": result,
            }
        )
        print("KT_ROOT_PROFILE_REQUEST", item["ordinal"], item["role"], flush=True)
    payload = {
        "schema": 1,
        "status": "PASS_KT_ROOT_CAUSE_PROFILE_REQUESTS",
        "block": args.block,
        "block_id": block_id,
        "started_at": started,
        "ended_at": time.time(),
        "generation_attempts": len(rows),
        "requests": rows,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(payload["status"], flush=True)


if __name__ == "__main__":
    main()
