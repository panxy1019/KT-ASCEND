"""CPU-only parser test on existing smoke prefill/lookahead captures, not FD."""
import argparse
import json
from pathlib import Path
import torch
from run_free_generation_captured import inspect_capture


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smoke-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("refuse parser test overwrite")
    smoke = json.loads((args.smoke_root / "smoke_result.json").read_text())
    captures = {}
    for path in (args.smoke_root / "tensors").rglob("Pass*.pt"):
        tx = torch.load(path, map_location="cpu", weights_only=False)["capture.transaction"]
        captures.setdefault(tx["batch"]["request_ids"][0], []).append((tx["forward_sequence_id"], path, tx))
    count = 0
    for row in smoke["rows"]:
        request = row["request"]
        files = sorted(captures[request["request_id"]])
        if len(files) != 2:
            raise ValueError("SMOKE_TARGET_LOOKAHEAD_COUNT_MISMATCH")
        for step, (_, path, tx) in enumerate(files):
            inspect_capture(path, dict(input_ids=request["input_ids"]), [row["output_token"]], step,
                            request["request_id"], tx["provenance"], smoke["mode"])
            count += 1
    report = dict(status="PASS", mode=smoke["mode"], checked_forwards=count,
                  scope="CPU parser replay of existing smoke; NOT free-generation qualification",
                  new_model_requests=0)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
