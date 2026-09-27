"""Retire only unreferenced Round10 model/bank file cache with POSIX_FADV_DONTNEED."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time


SUFFIXES = {".gguf", ".safetensors"}


def _under(path, roots):
    try:
        resolved = Path(path).resolve(strict=False)
    except (OSError, RuntimeError):
        return False
    return any(resolved == root or root in resolved.parents for root in roots)


def _targeted(path, roots, exact_files):
    try:
        resolved = Path(path).resolve(strict=False)
    except (OSError, RuntimeError):
        return False
    return resolved in exact_files or any(
        resolved == root or root in resolved.parents for root in roots
    )


def scan_process_references(roots, proc_root=Path("/proc"), exact_files=(), pids=None):
    roots = [Path(root).resolve() for root in roots]
    exact_files = {Path(path).resolve(strict=False) for path in exact_files}
    found = []
    processes = (
        [proc_root / str(pid) for pid in pids]
        if pids is not None
        else proc_root.glob("[0-9]*")
    )
    for process in processes:
        try:
            pid = int(process.name)
        except ValueError:
            continue
        maps = process / "maps"
        try:
            lines = maps.read_text(errors="replace").splitlines()
        except (FileNotFoundError, ProcessLookupError):
            lines = []
        except PermissionError as error:
            raise RuntimeError(f"cannot audit maps for PID {pid}") from error
        for line in lines:
            fields = line.split(maxsplit=5)
            if (
                len(fields) == 6
                and fields[5].startswith("/")
                and _targeted(fields[5].removesuffix(" (deleted)"), roots, exact_files)
            ):
                found.append({"pid": pid, "kind": "map", "path": fields[5]})
        fd_root = process / "fd"
        try:
            descriptors = list(fd_root.iterdir())
        except (FileNotFoundError, ProcessLookupError):
            descriptors = []
        except PermissionError as error:
            raise RuntimeError(f"cannot audit file descriptors for PID {pid}") from error
        for descriptor in descriptors:
            try:
                target = os.readlink(descriptor)
            except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
                continue
            if target.startswith("/") and _targeted(
                target.removesuffix(" (deleted)"), roots, exact_files
            ):
                found.append({"pid": pid, "kind": "fd", "fd": descriptor.name, "path": target})
    return found


def stat_identity(paths):
    rows = []
    for path in sorted(map(Path, paths)):
        stat = path.stat()
        rows.append(
            {
                "path": str(path),
                "device": stat.st_dev,
                "inode": stat.st_ino,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return rows


def identity_sha256(rows):
    raw = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def numa_meminfo(nodes=(2, 3, 4, 7)):
    result = {}
    for node in nodes:
        rows = {}
        for line in Path(f"/sys/devices/system/node/node{node}/meminfo").read_text().splitlines():
            values = line.split()
            if len(values) >= 4 and values[0] == "Node":
                rows[values[2].rstrip(":")] = int(values[3])
        result[str(node)] = {
            key: rows[key]
            for key in ("MemFree", "FilePages", "AnonPages", "FileHugePages")
        }
    return result


def system_available_kib():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1])
    raise RuntimeError("system MemAvailable missing")


def retire(roots, output, *, expected_file_count):
    roots = [Path(root).resolve(strict=True) for root in roots]
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    paths = sorted(
        path
        for root in roots
        for path in root.rglob("*")
        if path.is_file() and path.suffix in SUFFIXES
    )
    if len(paths) != expected_file_count:
        raise RuntimeError(
            f"target file identity changed: expected {expected_file_count}, got {len(paths)}"
        )
    before_stats = stat_identity(paths)
    before_references = scan_process_references(roots, exact_files=paths)
    if before_references:
        raise RuntimeError(f"targeted file cache still has active references: {before_references[:8]}")
    report = {
        "schema": 1,
        "status": "RUNNING",
        "started": time.time(),
        "operation": "POSIX_FADV_DONTNEED",
        "global_drop_caches": False,
        "roots": [str(root) for root in roots],
        "suffixes": sorted(SUFFIXES),
        "file_count": len(paths),
        "file_stat_identity_sha256": identity_sha256(before_stats),
        "active_references_before": before_references,
        "before_numa_meminfo_kib": numa_meminfo(),
        "before_system_memavailable_kib": system_available_kib(),
        "failures": [],
    }
    for path in paths:
        try:
            descriptor = os.open(path, os.O_RDONLY)
            try:
                os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(descriptor)
        except BaseException as error:
            report["failures"].append({"path": str(path), "error": repr(error)})
    time.sleep(2)
    after_references = scan_process_references(roots, exact_files=paths)
    after_stats = stat_identity(paths)
    report.update(
        ended=time.time(),
        status="PASS_TARGETED_OWNED_FILE_CACHE_RETIREMENT"
        if not report["failures"] and after_stats == before_stats and not after_references
        else "FAIL",
        file_stats_unchanged=after_stats == before_stats,
        active_references_after=after_references,
        after_numa_meminfo_kib=numa_meminfo(),
        after_system_memavailable_kib=system_available_kib(),
        other_tasks_modified=False,
        automation="PAUSED_UNCHANGED",
    )
    output.write_text(json.dumps(report, indent=2) + "\n")
    if report["status"] != "PASS_TARGETED_OWNED_FILE_CACHE_RETIREMENT":
        raise RuntimeError("targeted file cache retirement failed")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-file-count", type=int, required=True)
    args = parser.parse_args()
    report = retire(args.root, args.output, expected_file_count=args.expected_file_count)
    print(report["status"], report["file_count"], flush=True)


if __name__ == "__main__":
    main()
