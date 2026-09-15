"""Strict defer0 self-history forcing parity, plus real forced-token injection."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--forced", type=Path, required=True)
    p.add_argument("--trace", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    a, b = json.loads(args.baseline.read_text()), json.loads(args.forced.read_text())
    if a["status"] != "PREQUALIFICATION_CAPTURE_PASS" or b["status"] != a["status"]:
        raise ValueError("R63_INCOMPLETE_CAPTURE")
    if a["forced"] or not b["forced"]:
        raise ValueError("R63_INVALID_PARITY_MODES")
    b_rows = {r["rid"]: r for r in b["rows"]}
    checked = 0
    for row in a["rows"]:
        other = b_rows[row["rid"]]
        if row["baseline_output_ids"] != other["baseline_output_ids"]:
            raise ValueError("R63_TOKEN_PARITY_FAILURE")
        for s, t in zip(row["steps"], other["steps"], strict=True):
            if s["pid"] == t["pid"]:
                raise ValueError("R63_REQUIRES_SEPARATE_PROCESS_CACHE")
            if s["logits_sha256"] != t["logits_sha256"]:
                raise ValueError("R63_LOGITS_PARITY_FAILURE")
            checked += 1
    trace = [json.loads(line) for line in args.trace.read_text().splitlines()]
    synthetic = [r for r in trace if r["rid"].endswith("-synthetic") and r["classification"] == "TARGET"]
    changed = sum(r["natural_token"] != r["returned_token"] for r in synthetic)
    if not synthetic or not changed or any(r["returned_token"] != 101 for r in synthetic):
        raise ValueError("R63_SYNTHETIC_CONTROL_NOT_EXERCISED")
    result = dict(status="PASS", exact_full_logit_forwards=checked,
                  separate_processes=True, synthetic_forced_targets=len(synthetic),
                  synthetic_non_greedy_injections=changed,
                  scope="defer0 self-history forcing parity; not numerical qualification")
    with args.output.open("x") as out:
        json.dump(result, out, indent=2)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
