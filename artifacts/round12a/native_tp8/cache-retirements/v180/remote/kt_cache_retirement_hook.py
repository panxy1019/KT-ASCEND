"""Fail-closed retirement hook for registered KT project weight-file cache.

The hook operates on an exact, pre-verified manifest.  It never discovers
candidate files recursively and never uses the global drop_caches interface.
"""

from __future__ import annotations

import argparse
import errno
import json
import os
from pathlib import Path
import stat as stat_module
import subprocess
import time
from typing import Callable, NamedTuple, Sequence
import uuid


ALLOWED_SUFFIXES = frozenset({".gguf", ".safetensors"})
DEFAULT_BANK_ROOT = Path("/mnt/data1/kt-round9/banks")
DEFAULT_NUMA_ROOT = Path("/sys/devices/system/node")


class RetirementRefused(RuntimeError):
    """A precondition or safety audit refused cache retirement."""


class RetirementFailed(RuntimeError):
    """The targeted operation ran but could not be verified complete."""


class TargetRecord(NamedTuple):
    scope: str
    alias_path: Path
    resolved_path: Path
    device: int
    inode: int
    size: int
    mtime_ns: int


class AuditResult(NamedTuple):
    complete: bool
    references: list[dict]
    errors: list[dict]


class ActivityResult(NamedTuple):
    complete: bool
    containers: list[dict]
    processes: list[dict]
    errors: list[dict]


def _identity(path: Path) -> tuple[int, int, int, int]:
    value = path.stat()
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns


def _record_identity(record: TargetRecord) -> tuple[int, int, int, int]:
    return record.device, record.inode, record.size, record.mtime_ns


def _path_inside(path: Path, root: Path) -> bool:
    return path != root and root in path.parents


def _registered_file(item: dict, scope: str, root: Path) -> tuple[Path, Path]:
    path_value = item.get("path")
    if not isinstance(path_value, str) or not path_value:
        raise RetirementRefused("registered file path is missing")
    supplied = Path(path_value)
    if not supplied.is_absolute():
        raise RetirementRefused("registered file path must be absolute")
    if Path(os.path.abspath(supplied)) != supplied:
        raise RetirementRefused(f"registered file path is not canonical: {supplied}")
    try:
        resolved = supplied.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise RetirementRefused(f"registered file cannot be resolved: {supplied}") from error
    try:
        declared_resolved = Path(item["resolved_path"])
    except (KeyError, TypeError) as error:
        raise RetirementRefused(f"registered resolved_path is missing: {supplied}") from error
    try:
        declared_is_canonical = (
            declared_resolved.is_absolute()
            and declared_resolved.resolve(strict=True) == declared_resolved
        )
    except (OSError, RuntimeError) as error:
        raise RetirementRefused(
            f"registered resolved_path cannot be resolved: {declared_resolved}"
        ) from error
    if not declared_is_canonical:
        raise RetirementRefused(f"registered resolved_path is not canonical: {declared_resolved}")
    if declared_resolved != resolved:
        raise RetirementRefused(f"registered resolved_path does not match alias target: {supplied}")
    try:
        alias_parent = supplied.parent.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise RetirementRefused(f"registered alias parent cannot be resolved: {supplied}") from error
    alias_location = alias_parent / supplied.name
    if not _path_inside(alias_location, root) or not _path_inside(resolved, root):
        raise RetirementRefused(f"registered file escapes its allowed root: {resolved}")
    if supplied.suffix not in ALLOWED_SUFFIXES or resolved.suffix not in ALLOWED_SUFFIXES:
        raise RetirementRefused(f"registered file extension is not allowed: {supplied.suffix}")
    if scope == "checkpoint" and supplied != resolved:
        raise RetirementRefused(f"checkpoint symlink is not allowed: {supplied}")
    value = resolved.stat()
    if not stat_module.S_ISREG(value.st_mode):
        raise RetirementRefused(f"registered target is not a regular file: {resolved}")
    return supplied, resolved


def validate_manifest(
    manifest: dict,
    *,
    expected_bank_root: Path = DEFAULT_BANK_ROOT,
    expected_checkpoint_root: Path,
    expected_checkpoint_identity: str,
) -> list[TargetRecord]:
    """Validate teardown gates and return the exact registered target set."""
    if manifest.get("schema") != 1:
        raise RetirementRefused("unsupported retirement manifest schema")
    if manifest.get("experiment_block_complete") is not True:
        raise RetirementRefused("experiment block is not complete")
    if manifest.get("project_containers_running") != 0:
        raise RetirementRefused("project container is still running")
    if manifest.get("project_processes_running") != 0:
        raise RetirementRefused("project process is still running")
    if manifest.get("read_only_use") is not True:
        raise RetirementRefused("read-only use contract is not confirmed")
    verification = manifest.get("hash_verification", {})
    if not (
        verification.get("status") == "PASS"
        and verification.get("complete") is True
        and verification.get("full_file") is True
    ):
        raise RetirementRefused("full-file hash verification is incomplete")

    try:
        bank_root = Path(manifest["bank_root"]).resolve(strict=True)
        required_bank_root = Path(expected_bank_root).resolve(strict=True)
    except (KeyError, OSError, RuntimeError, TypeError) as error:
        raise RetirementRefused("bank root identity is unavailable") from error
    if bank_root != required_bank_root:
        raise RetirementRefused("bank root differs from the fixed KT bank root")

    checkpoint = manifest.get("checkpoint", {})
    try:
        checkpoint_root = Path(checkpoint["root"]).resolve(strict=True)
        required_checkpoint_root = Path(expected_checkpoint_root).resolve(strict=True)
    except (KeyError, OSError, RuntimeError, TypeError) as error:
        raise RetirementRefused("checkpoint root identity is unavailable") from error
    if checkpoint_root != required_checkpoint_root:
        raise RetirementRefused("checkpoint root differs from the expected current checkpoint root")
    if checkpoint.get("identity") != expected_checkpoint_identity:
        raise RetirementRefused("checkpoint identity differs from the expected current checkpoint identity")

    activity = manifest.get("project_activity")
    if not isinstance(activity, dict):
        raise RetirementRefused("project activity contract is missing")
    for key in ("container_names", "container_id_prefixes", "process_markers"):
        values = activity.get(key)
        if not isinstance(values, list) or any(not isinstance(value, str) or not value for value in values):
            raise RetirementRefused(f"project activity {key} must be an exact string list")

    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise RetirementRefused("exact registered file set is empty")
    records: list[TargetRecord] = []
    seen_paths: set[Path] = set()
    seen_inodes: set[tuple[int, int]] = set()
    scopes: set[str] = set()
    for item in files:
        if not isinstance(item, dict):
            raise RetirementRefused("registered file row is malformed")
        scope = item.get("scope")
        if scope not in {"bank", "checkpoint"}:
            raise RetirementRefused("registered file scope is invalid")
        scopes.add(scope)
        if item.get("read_only_registered") is not True:
            raise RetirementRefused("file read-only registration is not confirmed")
        root = bank_root if scope == "bank" else checkpoint_root
        alias_path, resolved_path = _registered_file(item, scope, root)
        digest = item.get("sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or item.get("full_file_sha256_verified") is not True
        ):
            raise RetirementRefused(f"file hash verification is incomplete: {alias_path}")
        identity = item.get("stat")
        if not isinstance(identity, dict):
            raise RetirementRefused(f"stat identity is missing: {alias_path}")
        try:
            record = TargetRecord(
                scope,
                alias_path,
                resolved_path,
                int(identity["device"]),
                int(identity["inode"]),
                int(identity["size"]),
                int(identity["mtime_ns"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise RetirementRefused(f"stat identity is malformed: {alias_path}") from error
        if _identity(resolved_path) != _record_identity(record):
            raise RetirementRefused(f"stat identity changed before retirement: {alias_path}")
        inode = (record.device, record.inode)
        if alias_path in seen_paths or inode in seen_inodes:
            raise RetirementRefused(f"duplicate registered file identity: {alias_path}")
        seen_paths.add(alias_path)
        seen_inodes.add(inode)
        records.append(record)
    if scopes != {"bank", "checkpoint"}:
        raise RetirementRefused("exact file set must include bank and checkpoint files")
    return sorted(records, key=lambda row: str(row.alias_path))


def audit_proc_references(
    records: Sequence[TargetRecord],
    *,
    proc_root: Path = Path("/proc"),
    pids: Sequence[int] | None = None,
    scandir: Callable = os.scandir,
) -> AuditResult:
    """Audit every fd and map_files entry by target device and inode."""
    targets = {
        (row.device, row.inode): {
            "alias_path": str(row.alias_path),
            "resolved_path": str(row.resolved_path),
        }
        for row in records
    }
    if pids is None:
        try:
            processes = sorted(
                (entry for entry in Path(proc_root).iterdir() if entry.name.isdigit()),
                key=lambda entry: int(entry.name),
            )
        except OSError as error:
            return AuditResult(False, [], [{"pid": None, "kind": "proc", "error": repr(error)}])
    else:
        processes = [Path(proc_root) / str(pid) for pid in pids]

    references: list[dict] = []
    errors: list[dict] = []
    seen: set[tuple[int, str, int, int]] = set()
    for process in processes:
        pid = int(process.name)
        for kind, directory_name in (("fd", "fd"), ("map_file", "map_files")):
            directory = process / directory_name
            try:
                entries = list(scandir(directory))
            except (FileNotFoundError, ProcessLookupError):
                continue
            except OSError as error:
                errors.append({"pid": pid, "kind": kind, "error": repr(error)})
                continue
            for entry in entries:
                entry_path = Path(entry.path) if hasattr(entry, "path") else Path(entry)
                try:
                    value = entry.stat() if hasattr(entry, "stat") else entry_path.stat()
                except OSError as error:
                    if error.errno in {errno.ENOENT, errno.ESRCH}:
                        continue
                    errors.append(
                        {"pid": pid, "kind": kind, "entry": entry_path.name, "error": repr(error)}
                    )
                    continue
                key = (value.st_dev, value.st_ino)
                if key not in targets or (pid, kind, *key) in seen:
                    continue
                seen.add((pid, kind, *key))
                references.append(
                    {
                        "pid": pid,
                        "kind": kind,
                        "entry": entry_path.name,
                        **targets[key],
                        "device": key[0],
                        "inode": key[1],
                    }
                )
    return AuditResult(not errors, references, errors)


def resident_pages(path: Path) -> int:
    """Return resident pages using fincore's byte count."""
    raw = subprocess.check_output(["fincore", "-J", "-b", str(path)], text=True)
    rows = json.loads(raw).get("fincore", [])
    if len(rows) != 1 or "res" not in rows[0]:
        raise RetirementFailed(f"fincore returned no unique resident byte count for {path}")
    resident_bytes = int(rows[0]["res"])
    page_size = os.sysconf("SC_PAGE_SIZE")
    return (resident_bytes + page_size - 1) // page_size


def advise_dontneed(record: TargetRecord) -> None:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(record.resolved_path, flags)
    try:
        value = os.fstat(descriptor)
        opened = (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)
        if opened != _record_identity(record):
            raise RetirementRefused(
                f"opened file identity differs from registration: {record.resolved_path}"
            )
        os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(descriptor)


def read_numa_memfree(numa_root: Path = DEFAULT_NUMA_ROOT) -> dict[str, int]:
    result: dict[str, int] = {}
    for node in sorted(Path(numa_root).glob("node[0-9]*")):
        value = None
        for line in (node / "meminfo").read_text().splitlines():
            fields = line.split()
            if len(fields) >= 4 and fields[0] == "Node" and fields[2] == "MemFree:":
                value = int(fields[3])
                break
        if value is None:
            raise RetirementFailed(f"NUMA MemFree is unavailable for {node.name}")
        result[node.name.removeprefix("node")] = value
    if not result:
        raise RetirementFailed("no NUMA MemFree nodes were readable")
    return result


def audit_project_activity(
    contract: dict,
    *,
    proc_root: Path = Path("/proc"),
    command: Callable = subprocess.check_output,
) -> ActivityResult:
    """Recheck exact owned container selectors and process command markers."""
    containers: list[dict] = []
    processes: list[dict] = []
    errors: list[dict] = []
    names = set(contract["container_names"])
    prefixes = tuple(contract["container_id_prefixes"])
    if names or prefixes:
        try:
            raw = command(
                ["docker", "ps", "--no-trunc", "--format", "{{.ID}}\t{{.Names}}"],
                text=True,
                timeout=30,
            )
        except Exception as error:
            errors.append({"kind": "container_audit", "error": repr(error)})
        else:
            for line in raw.splitlines():
                container_id, separator, name = line.partition("\t")
                if not separator:
                    errors.append({"kind": "container_audit", "error": "malformed docker ps row"})
                    continue
                if name in names or any(container_id.startswith(prefix) for prefix in prefixes):
                    containers.append({"id": container_id, "name": name})

    markers = tuple(contract["process_markers"])
    if markers:
        try:
            proc_entries = list(Path(proc_root).iterdir())
        except OSError as error:
            errors.append({"kind": "process_audit", "error": repr(error)})
            proc_entries = []
        for process in proc_entries:
            if not process.name.isdigit():
                continue
            try:
                raw = (process / "cmdline").read_bytes()
            except (FileNotFoundError, ProcessLookupError):
                continue
            except OSError as error:
                errors.append(
                    {"kind": "process_audit", "pid": int(process.name), "error": repr(error)}
                )
                continue
            command_line = raw.replace(b"\0", b" ").decode(errors="replace")
            matched = [marker for marker in markers if marker in command_line]
            if matched:
                processes.append(
                    {"pid": int(process.name), "markers": matched, "cmdline": command_line}
                )
    return ActivityResult(not errors, containers, processes, errors)


class SystemDependencies:
    def project_activity(self, contract: dict) -> ActivityResult:
        return audit_project_activity(contract)

    def audit(self, records: Sequence[TargetRecord]) -> AuditResult:
        return audit_proc_references(records)

    def resident_pages(self, path: Path) -> int:
        return resident_pages(path)

    def fadvise(self, record: TargetRecord) -> None:
        advise_dontneed(record)

    def numa_memfree(self) -> dict[str, int]:
        return read_numa_memfree()

    def settle(self) -> None:
        time.sleep(2)


def _assert_audit_authorizes(result: AuditResult) -> None:
    if not result.complete:
        raise RetirementRefused(f"process reference audit is incomplete: {result.errors[:8]}")
    if result.references:
        raise RetirementRefused(f"active reference to registered target: {result.references[:8]}")


def _assert_project_inactive(result: ActivityResult) -> None:
    if not result.complete:
        raise RetirementRefused(f"live project activity audit is incomplete: {result.errors[:8]}")
    if result.containers:
        raise RetirementRefused(f"project container is still active: {result.containers[:8]}")
    if result.processes:
        raise RetirementRefused(f"project process is still active: {result.processes[:8]}")


def _assert_identities(records: Sequence[TargetRecord], stage: str) -> None:
    for record in records:
        try:
            alias_target = record.alias_path.resolve(strict=True)
        except (OSError, RuntimeError) as error:
            raise RetirementFailed(f"registered alias changed {stage}: {record.alias_path}") from error
        if alias_target != record.resolved_path:
            raise RetirementFailed(f"registered alias changed {stage}: {record.alias_path}")
        if _identity(record.resolved_path) != _record_identity(record):
            raise RetirementFailed(f"stat identity changed {stage}: {record.resolved_path}")


def retire_project_cache(
    manifest: dict,
    *,
    deps=None,
    expected_bank_root: Path = DEFAULT_BANK_ROOT,
    expected_checkpoint_root: Path,
    expected_checkpoint_identity: str,
    dry_run: bool = False,
) -> dict:
    """Retire the exact audited target set, or return an explicit zero-page no-op."""
    dependencies = deps or SystemDependencies()
    records = validate_manifest(
        manifest,
        expected_bank_root=expected_bank_root,
        expected_checkpoint_root=expected_checkpoint_root,
        expected_checkpoint_identity=expected_checkpoint_identity,
    )
    activity = dependencies.project_activity(manifest["project_activity"])
    _assert_project_inactive(activity)
    audit_before = dependencies.audit(records)
    _assert_audit_authorizes(audit_before)
    before_numa = dependencies.numa_memfree()
    before = {
        str(row.resolved_path): int(dependencies.resident_pages(row.resolved_path))
        for row in records
    }
    if any(value < 0 for value in before.values()):
        raise RetirementFailed("fincore returned a negative resident page count")

    base = {
        "schema": 1,
        "operation": "POSIX_FADV_DONTNEED",
        "global_drop_caches": False,
        "dry_run": dry_run,
        "checkpoint_identity": manifest["checkpoint"]["identity"],
        "file_count": len(records),
        "files": [
            {"scope": row.scope, "alias_path": str(row.alias_path), "resolved_path": str(row.resolved_path)}
            for row in records
        ],
        "registered_stat_identity": [
            {
                "scope": row.scope,
                "alias_path": str(row.alias_path),
                "resolved_path": str(row.resolved_path),
                "device": row.device,
                "inode": row.inode,
                "size": row.size,
                "mtime_ns": row.mtime_ns,
            }
            for row in records
        ],
        "resident_pages_before": before,
        "numa_memfree_kib_before": before_numa,
        "audit_complete": True,
        "active_references": [],
    }
    if dry_run:
        return {**base, "status": "DRY_RUN_READY", "cache_retirement_performed": False}

    if sum(before.values()) == 0:
        _assert_identities(records, "during zero-residency verification")
        activity_after = dependencies.project_activity(manifest["project_activity"])
        _assert_project_inactive(activity_after)
        audit_after = dependencies.audit(records)
        _assert_audit_authorizes(audit_after)
        return {
            **base,
            "status": "NOOP_ZERO_RESIDENCY",
            "cache_retirement_performed": False,
            "cache_retirement_solved_admission": False,
            "resident_pages_after": dict(before),
            "numa_memfree_kib_after": dependencies.numa_memfree(),
            "stat_identity_unchanged": True,
        }

    _assert_identities(records, "immediately before cache retirement")
    activity_pre_mutation = dependencies.project_activity(manifest["project_activity"])
    _assert_project_inactive(activity_pre_mutation)
    audit_pre_mutation = dependencies.audit(records)
    _assert_audit_authorizes(audit_pre_mutation)
    failures = []
    for record in records:
        try:
            dependencies.fadvise(record)
        except Exception as error:
            failures.append({"path": str(record.resolved_path), "error": repr(error)})
    dependencies.settle()
    after = {
        str(row.resolved_path): int(dependencies.resident_pages(row.resolved_path))
        for row in records
    }
    after_numa = dependencies.numa_memfree()
    _assert_identities(records, "after cache retirement")
    activity_after = dependencies.project_activity(manifest["project_activity"])
    _assert_project_inactive(activity_after)
    audit_after = dependencies.audit(records)
    _assert_audit_authorizes(audit_after)
    if failures:
        raise RetirementFailed(f"POSIX_FADV_DONTNEED failed: {failures[:8]}")
    remaining = {path: pages for path, pages in after.items() if pages != 0}
    if remaining:
        raise RetirementFailed(f"resident pages remain after cache retirement: {remaining}")
    return {
        **base,
        "status": "PASS_CACHE_RETIRED",
        "cache_retirement_performed": True,
        "resident_pages_after": after,
        "numa_memfree_kib_after": after_numa,
        "stat_identity_unchanged": True,
    }


def retry_admission_after_cache_retirement(
    initial_admission: dict,
    retire: Callable[[], dict],
    admit: Callable[[str], dict],
    new_identity: Callable[[], str] = lambda: str(uuid.uuid4()),
) -> dict:
    """Perform at most one fresh admission retry after a verified retirement."""
    if initial_admission.get("fresh_admission") is not True:
        return {"status": "NO_RETRY_NOT_FRESH", "attempts": 0}
    if not (
        initial_admission.get("status") == "RESOURCE_INSUFFICIENT"
        and initial_admission.get("project_file_cache_cause") is True
    ):
        return {"status": "NO_RETRY_CAUSE_NOT_CONFIRMED", "attempts": 0}
    retirement = retire()
    if retirement.get("status") == "NOOP_ZERO_RESIDENCY":
        return {"status": "NO_RETRY_ZERO_RESIDENCY", "attempts": 0, "retirement": retirement}
    if retirement.get("status") != "PASS_CACHE_RETIRED":
        raise RetirementFailed("cache retirement did not authorize an admission retry")
    identity = new_identity()
    if not isinstance(identity, str) or not identity or identity == initial_admission.get("identity"):
        raise RetirementFailed("fresh admission identity was not generated")
    admission = admit(identity)
    return {
        "status": "ADMISSION_RETRY_COMPLETE",
        "attempts": 1,
        "retry_kind": "admission",
        "model_component_retry_increment": 0,
        "initial_identity": initial_admission.get("identity"),
        "retry_identity": identity,
        "retirement": retirement,
        "admission": admission,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-root", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-identity", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = json.loads(args.manifest.read_text())
    report = retire_project_cache(
        manifest,
        expected_checkpoint_root=args.expected_checkpoint_root,
        expected_checkpoint_identity=args.expected_checkpoint_identity,
        dry_run=args.dry_run,
    )
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(report["status"], report["file_count"], flush=True)


if __name__ == "__main__":
    main()
