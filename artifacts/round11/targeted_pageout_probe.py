"""Probe MADV_PAGEOUT on one audited, unreferenced project weight file."""

from __future__ import annotations

import argparse
import ctypes
import json
import mmap
import os
from pathlib import Path
import subprocess
import time


MADV_PAGEOUT = 21


def stat_identity(path):
    value = path.stat()
    return {
        "device": value.st_dev,
        "inode": value.st_ino,
        "size": value.st_size,
        "mtime_ns": value.st_mtime_ns,
    }


def resident_bytes(path):
    raw = subprocess.check_output(["fincore", "-J", "-b", str(path)], text=True)
    row = json.loads(raw)["fincore"][0]
    return int(row["res"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    path = args.file.resolve(strict=True)
    audit = json.loads(args.audit.read_text())
    if audit.get("status") != "PASS_AUDIT" or audit.get("references") != []:
        raise RuntimeError("inode reference audit does not authorize a pageout probe")
    before_stat = stat_identity(path)
    before_resident = resident_bytes(path)
    descriptor = os.open(path, os.O_RDONLY)
    mapping = mmap.mmap(descriptor, 0, access=mmap.ACCESS_COPY)
    try:
        address = ctypes.addressof(ctypes.c_char.from_buffer(mapping))
        libc = ctypes.CDLL(None, use_errno=True)
        result = libc.madvise(ctypes.c_void_p(address), ctypes.c_size_t(len(mapping)), MADV_PAGEOUT)
        error_number = ctypes.get_errno()
    finally:
        mapping.close()
        os.close(descriptor)
    time.sleep(2)
    after_resident = resident_bytes(path)
    after_stat = stat_identity(path)
    report = {
        "schema": 1,
        "status": "PASS_TARGETED_PAGEOUT_PROBE" if result == 0 and before_stat == after_stat else "FAIL",
        "file": str(path),
        "madvise": "MADV_PAGEOUT",
        "global_drop_caches": False,
        "active_references": 0,
        "result": result,
        "errno": error_number,
        "before_resident_bytes": before_resident,
        "after_resident_bytes": after_resident,
        "file_stat_unchanged": before_stat == after_stat,
        "other_tasks_modified": False,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if report["status"] != "PASS_TARGETED_PAGEOUT_PROBE":
        raise RuntimeError("targeted pageout probe failed")
    print(report["status"], before_resident, after_resident, flush=True)


if __name__ == "__main__":
    main()
