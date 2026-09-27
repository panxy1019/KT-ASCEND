"""Run the frozen eight-generation Round10 final configuration schedule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import round10_t1_driver as t1


def run(out, fixture):
    original = t1.build_schedule

    def final_schedule(document, process):
        if process != "F":
            raise ValueError("T6 final process identity required")
        return original(document, "R")[:8]

    t1.build_schedule = final_schedule
    try:
        report = t1.run("F", out, fixture)
    finally:
        t1.build_schedule = original
    if report.get("generation_attempts") != 8 or report.get("abort_controls") != 1:
        raise RuntimeError("T6 final schedule budget mismatch")
    report["status"] = "PASS_ROUND10_T6_PROCESS"
    report["baseline"] = "HFINAL_EQUALS_H0_ROUND10"
    source = Path(out) / "round10-t1-results.json"
    target = Path(out) / "round10-t6-results.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--process", choices=("F",), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.out, args.fixture)
    print(report["status"], args.process, flush=True)


if __name__ == "__main__":
    main()
