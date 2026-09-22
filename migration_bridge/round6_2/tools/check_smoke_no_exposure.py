"""Offline control over already captured smoke inputs; no numerical fitting."""
import argparse
import json
from pathlib import Path
import torch
from assemble_capture import byte_sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--defer0", type=Path, required=True)
    parser.add_argument("--defer1", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refuse control overwrite")
    results = [json.loads((root / "smoke_result.json").read_text()) for root in (args.defer0, args.defer1)]
    logits = [torch.load(root / "smoke_logits.pt", weights_only=True, map_location="cpu") for root in (args.defer0, args.defer1)]
    if [r["mode"] for r in results] != ["defer0", "defer1"]:
        raise ValueError("CONTROL_MODE_MISMATCH")
    rows = []
    for index, (base, deferred) in enumerate(zip(results[0]["rows"], results[1]["rows"])):
        if base["request_input_hash"] != deferred["request_input_hash"]:
            raise ValueError("CONTROL_INPUT_MISMATCH")
        if deferred["deferred_exposure"] != "NO_EXPOSURE":
            continue
        rows.append(dict(index=index, exact=byte_sha(logits[0][index]) == byte_sha(logits[1][index]),
                         max_abs=float((logits[0][index].double()-logits[1][index].double()).abs().max())))
    passed = bool(rows) and all(r["exact"] for r in rows)
    report = dict(status="PASS" if passed else "NO_EXPOSURE_LOGIT_MISMATCH", rows=rows, count=len(rows))
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
