"""Supplement frozen pilot's text audit with exact historical input-ID audit."""
import argparse
import hashlib
import json
from pathlib import Path


def inputs(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "input_ids" and isinstance(item, list) and item and all(type(t) is int for t in item):
                yield tuple(item)
            else:
                yield from inputs(item)
    elif isinstance(value, list):
        for item in value:
            yield from inputs(item)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pilot", type=Path, required=True)
    p.add_argument("--historical-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    historical, sources = set(), {}
    for path in sorted(args.historical_root.rglob("*.json")):
        if "round6_3" in path.parts or not ({"corpora", "fixtures"} & set(path.parts)):
            continue
        raw = path.read_bytes()
        values = list(inputs(json.loads(raw)))
        if values:
            historical.update(values)
            sources[str(path)] = hashlib.sha256(raw).hexdigest()
    if not historical:
        raise ValueError("NO_HISTORICAL_TOKEN_INPUTS")
    raw = args.pilot.read_bytes()
    pilot = json.loads(raw)
    duplicates = [r["rid"] for r in pilot["requests"] if tuple(r["input_ids"]) in historical]
    report = dict(status="FAIL" if duplicates else "PASS", duplicate_rids=duplicates,
        pilot_sha256=hashlib.sha256(raw).hexdigest(), historical_unique_inputs=len(historical), sources=sources)
    with args.output.open("x") as out:
        json.dump(report, out, indent=2)
    print(json.dumps({k:v for k,v in report.items() if k != "sources"}))
    if duplicates:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
