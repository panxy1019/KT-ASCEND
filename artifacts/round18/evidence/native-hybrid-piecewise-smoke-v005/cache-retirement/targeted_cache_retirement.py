"""Retire cache only for an exact, registered set of read-only model/bank files.

The manifest is prepared after the trusted full-file receipts have been checked.  This
program deliberately does not hash the large files again because doing so would warm
the same page cache that it is retiring.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Iterable


ALLOWED_SUFFIXES = {".gguf", ".safetensors"}


def parse_fincore_json(payload: str) -> list[dict]:
    document = json.loads(payload)
    source = document.get("fincore")
    if not isinstance(source, list):
        raise RuntimeError("fincore JSON is missing the fincore row list")
    rows = []
    for source_row in source:
        try:
            rows.append(
                {
                    "path": str(Path(source_row["file"]).resolve(strict=False)),
                    "size_bytes": int(source_row["size"]),
                    "resident_pages": int(source_row["pages"]),
                    "resident_bytes": int(source_row["res"]),
                }
            )
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(f"invalid fincore row: {source_row!r}") from error
    return rows


def summarize_fincore(rows: Iterable[dict]) -> dict:
    rows = list(rows)
    return {
        "file_count": len(rows),
        "total_size_bytes": sum(int(row["size_bytes"]) for row in rows),
        "resident_pages": sum(int(row["resident_pages"]) for row in rows),
        "resident_bytes": sum(int(row["resident_bytes"]) for row in rows),
        "resident_file_count": sum(int(row["resident_pages"]) > 0 for row in rows),
    }


def _stat_identity(path: Path) -> dict:
    stat = path.stat()
    return {
        "path": str(path),
        "size": stat.st_size,
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "mtime_ns": stat.st_mtime_ns,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_registered_manifest(
    *,
    bank_catalog: Path,
    bank_container_root: Path,
    bank_host_root: Path,
    model_index: Path,
    model_root: Path,
    expected_bank_count: int,
    expected_model_count: int,
) -> dict:
    bank_catalog = bank_catalog.resolve(strict=True)
    bank_host_root = bank_host_root.resolve(strict=True)
    model_index = model_index.resolve(strict=True)
    model_root = model_root.resolve(strict=True)
    container_root = bank_container_root.as_posix().rstrip("/") + "/"
    catalog = json.loads(bank_catalog.read_text())
    bank_rows = catalog.get("banks")
    if not isinstance(bank_rows, list) or len(bank_rows) != expected_bank_count:
        raise RuntimeError(
            f"bank catalog count mismatch: expected {expected_bank_count}, "
            f"got {len(bank_rows) if isinstance(bank_rows, list) else 'invalid'}"
        )
    targets = []
    for bank_row in bank_rows:
        source_path = str(bank_row["bank_path"])
        if not source_path.startswith(container_root):
            raise RuntimeError(f"bank path is outside the registered container root: {source_path}")
        relative = Path(source_path[len(container_root) :])
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"unsafe bank path: {source_path}")
        path = (bank_host_root / relative).resolve(strict=True)
        if bank_host_root != path and bank_host_root not in path.parents:
            raise RuntimeError(f"mapped bank path escaped the host root: {path}")
        identity = _stat_identity(path)
        if identity["size"] != int(bank_row["bytes"]):
            raise RuntimeError(f"bank catalog size mismatch: {path}")
        identity.update(role="bank", trusted_sha256=str(bank_row["sha256"]))
        targets.append(identity)
    model_document = json.loads(model_index.read_text())
    model_files = sorted(set(model_document.get("weight_map", {}).values()))
    if len(model_files) != expected_model_count:
        raise RuntimeError(
            f"model index count mismatch: expected {expected_model_count}, got {len(model_files)}"
        )
    for file_name in model_files:
        relative = Path(file_name)
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"unsafe model index path: {file_name}")
        path = (model_root / relative).resolve(strict=True)
        if model_root != path and model_root not in path.parents:
            raise RuntimeError(f"model path escaped the checkpoint root: {path}")
        identity = _stat_identity(path)
        identity.update(role="model")
        targets.append(identity)
    return {
        "schema": 1,
        "status": "REGISTERED_READ_ONLY_CACHE_TARGETS",
        "generated_unix": time.time(),
        "sources": {
            "bank_catalog": str(bank_catalog),
            "bank_catalog_sha256": _sha256_file(bank_catalog),
            "bank_container_root": str(bank_container_root),
            "bank_host_root": str(bank_host_root),
            "model_index": str(model_index),
            "model_index_sha256": _sha256_file(model_index),
            "model_root": str(model_root),
        },
        "counts": {"bank": len(bank_rows), "model": len(model_files)},
        "targets": targets,
    }


def validate_registered_targets(rows: list[dict], *, expected_count: int) -> list[Path]:
    if len(rows) != expected_count:
        raise RuntimeError(
            f"registered target count mismatch: expected {expected_count}, got {len(rows)}"
        )
    paths = []
    seen = set()
    for row in rows:
        path = Path(row["path"]).resolve(strict=True)
        if path in seen:
            raise RuntimeError(f"duplicate registered target: {path}")
        if path.suffix not in ALLOWED_SUFFIXES:
            raise RuntimeError(f"unapproved target suffix: {path}")
        if not path.is_file():
            raise RuntimeError(f"registered target is not a regular file: {path}")
        current = _stat_identity(path)
        expected = {
            key: row[key]
            for key in ("size", "device", "inode", "mtime_ns")
        }
        observed = {key: current[key] for key in expected}
        if observed != expected:
            raise RuntimeError(
                f"identity mismatch for {path}: expected {expected}, observed {observed}"
            )
        seen.add(path)
        paths.append(path)
    return sorted(paths)


def identity_sha256(paths: Iterable[Path]) -> str:
    rows = [_stat_identity(path) for path in sorted(paths)]
    raw = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _targeted(raw_path: str, targets: set[Path]) -> bool:
    clean = raw_path.removesuffix(" (deleted)")
    if not clean.startswith("/"):
        return False
    try:
        return Path(clean).resolve(strict=False) in targets
    except (OSError, RuntimeError):
        return False


def scan_process_references(
    paths: Iterable[Path], *, proc_root: Path = Path("/proc"), pids=None
) -> list[dict]:
    targets = {Path(path).resolve(strict=False) for path in paths}
    processes = (
        [proc_root / str(pid) for pid in pids]
        if pids is not None
        else sorted(proc_root.glob("[0-9]*"), key=lambda item: int(item.name))
    )
    found = []
    for process in processes:
        try:
            pid = int(process.name)
        except ValueError:
            continue
        try:
            map_lines = (process / "maps").read_text(errors="replace").splitlines()
        except (FileNotFoundError, ProcessLookupError):
            map_lines = []
        except PermissionError as error:
            raise RuntimeError(f"cannot audit maps for PID {pid}") from error
        for line in map_lines:
            fields = line.split(maxsplit=5)
            if len(fields) == 6 and _targeted(fields[5], targets):
                found.append({"pid": pid, "kind": "map", "path": fields[5]})
        try:
            descriptors = list((process / "fd").iterdir())
        except (FileNotFoundError, ProcessLookupError):
            descriptors = []
        except PermissionError as error:
            raise RuntimeError(f"cannot audit descriptors for PID {pid}") from error
        for descriptor in descriptors:
            try:
                target = os.readlink(descriptor)
            except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
                continue
            if _targeted(target, targets):
                found.append(
                    {
                        "pid": pid,
                        "kind": "fd",
                        "fd": descriptor.name,
                        "path": target,
                    }
                )
    return found


def run_fincore(paths: Iterable[Path], *, batch_size: int = 48) -> list[dict]:
    paths = list(paths)
    rows = []
    for start in range(0, len(paths), batch_size):
        batch = paths[start : start + batch_size]
        command = [
            "fincore",
            "--json",
            "--bytes",
            "--output",
            "PAGES,SIZE,RES,FILE",
            *map(str, batch),
        ]
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        rows.extend(parse_fincore_json(result.stdout))
    expected = {str(path.resolve(strict=False)) for path in paths}
    observed = {row["path"] for row in rows}
    if len(rows) != len(paths) or observed != expected:
        raise RuntimeError(
            f"fincore coverage mismatch: expected {len(expected)}, observed {len(observed)}"
        )
    return sorted(rows, key=lambda row: row["path"])


def retirement_sufficient(before: int, after: int, *, minimum_drop_bytes: int) -> bool:
    if before == 0:
        return after == 0
    required = min(before, minimum_drop_bytes)
    return after < before and before - after >= required


def numa_meminfo(nodes=(2, 3, 4, 7)) -> dict:
    result = {}
    for node in nodes:
        values = {}
        source = Path(f"/sys/devices/system/node/node{node}/meminfo")
        for line in source.read_text().splitlines():
            fields = line.split()
            if len(fields) >= 4 and fields[0] == "Node":
                values[fields[2].rstrip(":")] = int(fields[3])
        result[str(node)] = {
            key: values[key]
            for key in ("MemFree", "FilePages", "AnonPages", "FileHugePages")
        }
    return result


def system_available_kib() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1])
    raise RuntimeError("system MemAvailable is unavailable")


def _atomic_write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def retire(
    manifest_path: Path,
    output_path: Path,
    *,
    expected_count: int,
    minimum_drop_bytes: int,
) -> dict:
    if output_path.exists():
        raise FileExistsError(output_path)
    manifest = json.loads(manifest_path.read_text())
    report = {
        "schema": 2,
        "status": "RUNNING",
        "started_unix": time.time(),
        "operation": "POSIX_FADV_DONTNEED",
        "global_drop_caches": False,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "expected_file_count": expected_count,
        "minimum_drop_bytes": minimum_drop_bytes,
        "failures": [],
    }
    try:
        paths = validate_registered_targets(
            manifest["targets"], expected_count=expected_count
        )
        report["file_count"] = len(paths)
        report["roles"] = {
            role: sum(row["role"] == role for row in manifest["targets"])
            for role in sorted({row["role"] for row in manifest["targets"]})
        }
        report["file_stat_identity_sha256"] = identity_sha256(paths)
        report["before_numa_meminfo_kib"] = numa_meminfo()
        report["before_system_memavailable_kib"] = system_available_kib()
        before_references = scan_process_references(paths)
        report["active_references_before"] = before_references
        if before_references:
            raise RuntimeError(
                f"registered files have active references: {before_references[:8]}"
            )
        before_rows = run_fincore(paths)
        report["before_fincore"] = summarize_fincore(before_rows)
        for path in paths:
            descriptor = os.open(path, os.O_RDONLY)
            try:
                os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(descriptor)
        time.sleep(2)
        after_references = scan_process_references(paths)
        report["active_references_after"] = after_references
        if after_references:
            raise RuntimeError(
                f"registered files gained active references: {after_references[:8]}"
            )
        if identity_sha256(paths) != report["file_stat_identity_sha256"]:
            raise RuntimeError("registered file identity changed during retirement")
        after_rows = run_fincore(paths)
        report["after_fincore"] = summarize_fincore(after_rows)
        report["resident_bytes_released"] = (
            report["before_fincore"]["resident_bytes"]
            - report["after_fincore"]["resident_bytes"]
        )
        report["after_numa_meminfo_kib"] = numa_meminfo()
        report["after_system_memavailable_kib"] = system_available_kib()
        if not retirement_sufficient(
            report["before_fincore"]["resident_bytes"],
            report["after_fincore"]["resident_bytes"],
            minimum_drop_bytes=minimum_drop_bytes,
        ):
            raise RuntimeError("fincore did not confirm the required resident-page release")
        report["status"] = "PASS_TARGETED_REGISTERED_FILE_CACHE_RETIREMENT"
    except BaseException as error:
        report["status"] = "FAIL_TARGETED_REGISTERED_FILE_CACHE_RETIREMENT"
        report["failures"].append(
            {"type": type(error).__name__, "message": str(error)}
        )
        raise
    finally:
        report["ended_unix"] = time.time()
        report["other_tasks_modified"] = False
        report["automation"] = "PAUSED_UNCHANGED"
        _atomic_write_json(output_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build", help="freeze exact registered targets")
    build_parser.add_argument("--bank-catalog", type=Path, required=True)
    build_parser.add_argument("--bank-container-root", type=Path, required=True)
    build_parser.add_argument("--bank-host-root", type=Path, required=True)
    build_parser.add_argument("--model-index", type=Path, required=True)
    build_parser.add_argument("--model-root", type=Path, required=True)
    build_parser.add_argument("--expected-bank-count", type=int, required=True)
    build_parser.add_argument("--expected-model-count", type=int, required=True)
    build_parser.add_argument("--output", type=Path, required=True)
    retire_parser = subparsers.add_parser("retire", help="audit and retire registered cache")
    retire_parser.add_argument("--manifest", type=Path, required=True)
    retire_parser.add_argument("--output", type=Path, required=True)
    retire_parser.add_argument("--expected-file-count", type=int, required=True)
    retire_parser.add_argument("--minimum-drop-bytes", type=int, default=1 << 30)
    arguments = parser.parse_args()
    if arguments.command == "build":
        if arguments.output.exists():
            raise FileExistsError(arguments.output)
        manifest = build_registered_manifest(
            bank_catalog=arguments.bank_catalog,
            bank_container_root=arguments.bank_container_root,
            bank_host_root=arguments.bank_host_root,
            model_index=arguments.model_index,
            model_root=arguments.model_root,
            expected_bank_count=arguments.expected_bank_count,
            expected_model_count=arguments.expected_model_count,
        )
        _atomic_write_json(arguments.output, manifest)
        print(
            manifest["status"],
            manifest["counts"]["bank"],
            manifest["counts"]["model"],
            flush=True,
        )
        return
    report = retire(
        arguments.manifest,
        arguments.output,
        expected_count=arguments.expected_file_count,
        minimum_drop_bytes=arguments.minimum_drop_bytes,
    )
    print(
        report["status"],
        report["file_count"],
        report["resident_bytes_released"],
        flush=True,
    )


if __name__ == "__main__":
    main()
