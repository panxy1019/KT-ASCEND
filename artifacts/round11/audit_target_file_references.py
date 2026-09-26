"""Audit target file references by device/inode across process namespaces."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


SUFFIXES = {".gguf", ".safetensors"}


def identity(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino


def target_files(roots):
    paths = sorted(
        path
        for root in roots
        for path in root.rglob("*")
        if path.is_file() and path.suffix in SUFFIXES
    )
    return paths, {identity(path): str(path.resolve()) for path in paths}


def audit(roots, proc_root=Path("/proc")):
    paths, targets = target_files(roots)
    found = []
    errors = []
    for process in proc_root.glob("[0-9]*"):
        try:
            pid = int(process.name)
        except ValueError:
            continue
        seen = set()
        for kind, directory in (("fd", process / "fd"), ("map_file", process / "map_files")):
            try:
                entries = list(directory.iterdir())
            except (FileNotFoundError, ProcessLookupError):
                continue
            except PermissionError as error:
                errors.append({"pid": pid, "kind": kind, "error": str(error)})
                continue
            for entry in entries:
                try:
                    key = identity(entry)
                except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
                    continue
                if key in targets and (kind, key) not in seen:
                    seen.add((kind, key))
                    try:
                        link = os.readlink(entry)
                    except OSError:
                        link = "UNAVAILABLE"
                    found.append(
                        {
                            "pid": pid,
                            "kind": kind,
                            "entry": entry.name,
                            "link": link,
                            "target": targets[key],
                            "device": key[0],
                            "inode": key[1],
                        }
                    )
    return {
        "schema": 1,
        "status": "PASS_AUDIT" if not errors else "INCOMPLETE_AUDIT",
        "target_path_count": len(paths),
        "target_identity_count": len(targets),
        "references": found,
        "permission_errors": errors,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = audit([root.resolve(strict=True) for root in args.root])
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(report["status"], len(report["references"]), flush=True)


if __name__ == "__main__":
    main()
