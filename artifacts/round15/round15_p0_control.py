#!/usr/bin/env python3
"""Pure control-plane checks shared by Round15 P0 controller and supervisor."""

from __future__ import annotations

import re
from pathlib import Path


TRAINING_ID = "0419bcfee46f317d6833e038c28e81cf3d3ecf47323a423062bac2764e0ab2ef"
TRAINING_MEMORY_LIMIT_MIB = 15360
_PROCESS = re.compile(
    r"Process id:(?P<pid>\d+).*?Process memory\(MB\):(?P<memory_mib>\d+)",
    re.DOTALL,
)


def read_proc_cgroup(pid):
    return Path(f"/proc/{int(pid)}/cgroup").read_text()


def _is_registered_training_cgroup(text):
    return re.search(
        rf"(?<![0-9a-f]){re.escape(TRAINING_ID)}(?![0-9a-f])", text
    ) is not None


def classify_physical12_after(proc_mem, cgroup_reader=read_proc_cgroup):
    """Allow an empty device or the one pre-registered training cgroup only."""
    if "No process in device" in proc_mem:
        return {"status": "EMPTY", "processes": []}
    rows = [
        {"pid": int(match.group("pid")), "memory_mib": int(match.group("memory_mib"))}
        for match in _PROCESS.finditer(proc_mem)
    ]
    if not rows:
        raise RuntimeError("unparseable physical12 process inventory")
    for row in rows:
        try:
            cgroup = cgroup_reader(row["pid"])
        except (FileNotFoundError, ProcessLookupError) as error:
            raise RuntimeError(
                f"physical12 process {row['pid']} disappeared before identity verification"
            ) from error
        if not _is_registered_training_cgroup(cgroup):
            raise RuntimeError(f"unknown physical12 process {row['pid']}")
        if row["memory_mib"] > TRAINING_MEMORY_LIMIT_MIB:
            raise RuntimeError(
                f"registered training process {row['pid']} exceeded memory contract"
            )
    return {"status": "REGISTERED_TRAINING_ONLY", "processes": rows}


def validate_supervisor_removed_container(supervisor, expected_container_id):
    """Prove a missing owned container was already retired by the supervisor."""
    ack = supervisor.get("guard_exit_ack")
    if (
        supervisor.get("container_id") != expected_container_id
        or supervisor.get("owned_container_removed") is not True
        or not isinstance(ack, dict)
        or not str(ack.get("action", "")).startswith("OWNED_EXITED")
        or not isinstance(ack.get("time"), (int, float))
    ):
        raise RuntimeError("missing owned container lacks exact supervisor removal proof")
    return {
        "status": "ALREADY_REMOVED_BY_SUPERVISOR",
        "container_id": expected_container_id,
        "supervisor_guard_exit_ack": ack,
    }
