#!/usr/bin/env python3
"""Run one immutable four-request Round15 promoted-S0 profile block."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


ROLES = {
    331: "A", 332: "B", 333: "A", 334: "B",
    335: "A", 336: "B",
    337: "A", 338: "B",
}


def _base():
    here = Path(__file__).resolve().parent
    candidates = (
        here / "round14a_b_profile_driver.py",
        here.parent / "round14a/round14a_b_profile_driver.py",
    )
    path = next((item for item in candidates if item.is_file()), None)
    if path is None:
        raise FileNotFoundError("Round15 P0 requires the frozen Round14a profile driver")
    spec = importlib.util.spec_from_file_location("_round15_p0_base_driver", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROLES = dict(ROLES)
    return module


def build_schedule(fixture, *, version, block):
    return _base().build_schedule(fixture, version=version, block=block)


def normalize_result(result):
    result.update(
        status="PASS_ROUND15_P0_PROFILE_DRIVER_RAW",
        stage="ROUND15_P0_FRESH_PROMOTED_S0_PROFILE",
        parent="ROUND14A_V295_PROMOTED_GDN_S0",
        profiler="BOUNDED_EXISTING_EVENT_OBSERVER",
        graph=False,
        round15_profile=True,
    )
    return result


def run(version, block, out, fixture_path):
    if ROLES.get(version) != block:
        raise ValueError("Round15 P0 role/version mismatch")
    out = Path(out)
    old = out / "round14a-b-profile-raw.json"
    target = out / "round15-p0-raw.json"
    if old.exists() or target.exists():
        raise FileExistsError("Round15 P0 result ledger already exists")
    base = _base()
    base.run(version, block, out, fixture_path)
    result = json.loads(old.read_text())
    if result.get("status") != "PASS_ROUND14A_B_PROFILE_DRIVER_RAW":
        raise RuntimeError("base profile driver did not complete")
    normalize_result(result)
    target.write_text(json.dumps(result, indent=2) + "\n")
    old.unlink()
    print(result["status"], version, block, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", type=int, choices=tuple(ROLES), required=True)
    parser.add_argument("--block", choices=tuple(ROLES.values()), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    args = parser.parse_args()
    run(args.version, args.block, args.out, args.fixture)


if __name__ == "__main__":
    main()
