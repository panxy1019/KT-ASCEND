"""After stopping own serving process, reconcile EVERY physical capture/index."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--server-root", type=Path, required=True)
    p.add_argument("--result", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    guard = json.loads((args.server_root / "guard/result.json").read_text())
    if guard["reason"] != "CANCELLED" or guard.get("matched_error"):
        raise ValueError("R63_SERVER_NOT_PLANNED_STOP")
    result = json.loads(args.result.read_text())
    entries = [json.loads(line) for line in (args.server_root / "index.jsonl").read_text().splitlines()]
    records = [s for row in result["rows"] for s in row["steps"]]
    if len(entries) != len(records):
        raise ValueError("R63_INDEX_COUNT")
    files = set((args.server_root / "tensors").rglob("Pass*.pt"))
    if files != {Path(e["file"]) for e in entries} or len(files) != len(entries):
        raise ValueError("R63_UNKNOWN_OR_DUPLICATE_PHYSICAL_CAPTURE")
    if list((args.server_root / "tensors").rglob("*.incomplete")):
        raise ValueError("R63_INCOMPLETE_FILE")
    if len({e["pid"] for e in entries}) != 1:
        raise ValueError("R63_RESTART_IN_CAPTURE")
    for i, (entry, record) in enumerate(zip(entries, records)):
        if entry["sequence"] != i or record["sequence"] != i or entry["file"] != record["file"] or entry["pid"] != record["pid"]:
            raise ValueError("R63_FINAL_INDEX_JOIN")
    report = dict(status="PASS", physical_captures=len(files), indexed=len(entries),
        targets=sum(s["target"] for s in records), lookahead=sum(not s["target"] for s in records),
        unknown=0, incomplete=0, numerical_metrics_read=False)
    with args.output.open("x") as out:
        json.dump(report, out, indent=2)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
