"""Pure validation for Round16's explicit NPU and NUMA admission target.

The historical admission program was tied to physical device 12 and PCI BDF
``0000:85:00.0``.  Round16 may use another device only when that exact device
identity and a reviewed four-node CPU profile are supplied explicitly.  This
module validates two already captured sampling documents.  It performs no
sampling, device initialization, process mutation, or cache eviction.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping, Sequence

from .resource_target import NpuTarget


_FULL_DOCKER_ID = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class AdmissionPolicy:
    target: NpuTarget
    training_container_id: str
    expected_host_ip: str
    expected_physical_cores: int = 160
    rounds: int = 2
    windows_per_round: int = 6
    minimum_round_seconds: float = 60.0
    minimum_idle_exclusive: float = 0.80
    minimum_node_free_kib: int = 48 * 1024**2
    minimum_system_available_kib: int = 704 * 1024**2
    expected_hbm_total_mib: int = 65536
    training_process_limit_mib: int = 15360

    def __post_init__(self) -> None:
        if _FULL_DOCKER_ID.fullmatch(self.training_container_id) is None:
            raise ValueError("training identity must be a full Docker ID")
        if not isinstance(self.expected_host_ip, str) or not self.expected_host_ip:
            raise ValueError("expected host IP is required")
        integer_fields = (
            self.expected_physical_cores,
            self.rounds,
            self.windows_per_round,
            self.minimum_node_free_kib,
            self.minimum_system_available_kib,
            self.expected_hbm_total_mib,
            self.training_process_limit_mib,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in integer_fields):
            raise ValueError("admission integer limits must be positive")
        if self.rounds != 2:
            raise ValueError("Round16 requires exactly two admission rounds")
        if self.minimum_round_seconds < 10:
            raise ValueError("each admission round must cover at least 10 seconds")
        if self.minimum_round_seconds < self.windows_per_round * 10:
            raise ValueError(
                "minimum round duration must cover every 10-second sampling window"
            )
        if not 0 < self.minimum_idle_exclusive < 1:
            raise ValueError("idle threshold must be between zero and one")


@dataclass(frozen=True)
class AdmissionResult:
    status: str
    target: NpuTarget
    numa_nodes: tuple[int, int, int, int]
    windows: int
    selected_physical_cores: int
    selected_parent_cpus: tuple[int, ...]
    min_mean_idle: float
    max_mean_idle: float
    window_means: tuple[float, ...]
    min_numa_free_gib: Mapping[int, float]


def _container_cgroup_matches(cgroup: object, container_id: str) -> bool:
    if not isinstance(cgroup, str):
        return False
    for line in cgroup.splitlines():
        path = line.split(":", 2)[-1]
        if path == f"/docker/{container_id}":
            return True
        if path.endswith(f"/docker-{container_id}.scope"):
            return True
    return False


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} is missing or malformed")
    return value


def _target_identity(document: Mapping[str, Any]) -> tuple[object, ...]:
    target = _mapping(document.get("target"), "target identity")
    nodes = target.get("numa_nodes")
    if not isinstance(nodes, (list, tuple)):
        raise RuntimeError("target identity has no explicit NUMA profile")
    return (
        target.get("physical_id"),
        target.get("card_id"),
        target.get("chip_id"),
        str(target.get("pci_bdf", "")).upper(),
        tuple(nodes),
    )


def _policy_target_identity(policy: AdmissionPolicy) -> tuple[object, ...]:
    target = policy.target
    return (
        target.physical_id,
        target.card_id,
        target.chip_id,
        target.pci_bdf,
        target.numa_nodes,
    )


def _selected_cores(document: Mapping[str, Any], policy: AdmissionPolicy) -> list[Mapping[str, Any]]:
    rows = document.get("cores")
    if not isinstance(rows, list):
        raise RuntimeError("candidate CPU topology is missing")
    result: list[Mapping[str, Any]] = []
    seen_siblings: set[int] = set()
    for value in rows:
        row = _mapping(value, "candidate CPU row")
        if row.get("node") not in policy.target.numa_nodes:
            continue
        if row.get("excluded_explicit_container_affinity") is not False:
            continue
        siblings = row.get("siblings")
        if not isinstance(siblings, list) or not siblings:
            raise RuntimeError("candidate CPU row has no SMT siblings")
        if any(isinstance(cpu, bool) or not isinstance(cpu, int) or cpu < 0 for cpu in siblings):
            raise RuntimeError("candidate CPU row has malformed SMT identity")
        if seen_siblings.intersection(siblings):
            raise RuntimeError("candidate CPU topology repeats a logical CPU")
        seen_siblings.update(siblings)
        result.append(row)
    if len(result) != policy.expected_physical_cores:
        raise RuntimeError(
            f"frozen {policy.expected_physical_cores}-core candidate set unavailable"
        )
    return result


def _core_identity(cores: Sequence[Mapping[str, Any]]) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            row.get("node"),
            row.get("socket"),
            row.get("core"),
            tuple(row.get("siblings", ())),
        )
        for row in cores
    )


def _conservative_mean(
    cores: Sequence[Mapping[str, Any]], idle: Mapping[str, Any]
) -> float:
    values: list[float] = []
    for row in cores:
        try:
            values.append(min(float(idle[str(cpu)]) for cpu in row["siblings"]))
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError("CPU idle sample is incomplete") from error
    return sum(values) / len(values)


def _validate_training_identity(value: object, policy: AdmissionPolicy) -> None:
    identity = _mapping(value, "training identity")
    if identity.get("id") != policy.training_container_id or identity.get("running") is not True:
        raise RuntimeError("training container identity changed")
    if not _container_cgroup_matches(identity.get("cgroup"), policy.training_container_id):
        raise RuntimeError("training container cgroup identity changed")


def _normalize_claims(value: object) -> dict[int, str]:
    claims = _mapping(value, "device claims")
    normalized: dict[int, str] = {}
    for key, owner in claims.items():
        try:
            physical_id = int(key)
        except (TypeError, ValueError) as error:
            raise RuntimeError("device claim has malformed physical ID") from error
        if not isinstance(owner, str) or not owner:
            raise RuntimeError("device claim has malformed owner")
        if physical_id in normalized:
            raise RuntimeError("device claim repeats a physical ID")
        normalized[physical_id] = owner
    return normalized


def _validate_device_window(
    window: Mapping[str, Any], policy: AdmissionPolicy
) -> None:
    claims = _normalize_claims(window.get("device_claims"))
    if policy.target.physical_id in claims:
        raise RuntimeError("target has an external container claim")

    device = _mapping(window.get("target_device"), "target device")
    actual_identity = (
        device.get("physical_id"),
        device.get("card_id"),
        device.get("chip_id"),
        str(device.get("pci_bdf", "")).upper(),
    )
    expected_identity = _policy_target_identity(policy)[:4]
    if actual_identity != expected_identity:
        raise RuntimeError("target device identity changed during admission")
    if device.get("hbm_total_mib") != policy.expected_hbm_total_mib:
        raise RuntimeError("target HBM total identity changed")

    processes = device.get("processes")
    if not isinstance(processes, list):
        raise RuntimeError("target process evidence is missing")
    for raw_process in processes:
        process = _mapping(raw_process, "target process")
        if not _container_cgroup_matches(
            process.get("cgroup"), policy.training_container_id
        ):
            raise RuntimeError(f"unknown target process {process.get('pid')}")
        memory = process.get("memory_mib")
        if isinstance(memory, bool) or not isinstance(memory, int) or memory < 0:
            raise RuntimeError("target process memory evidence is malformed")
        if memory > policy.training_process_limit_mib:
            raise RuntimeError("training process exceeds shared HBM allowance")


def validate_admission_documents(
    documents: Sequence[Mapping[str, Any]], policy: AdmissionPolicy
) -> AdmissionResult:
    """Validate exactly two immutable admission rounds, failing closed."""

    if len(documents) != policy.rounds:
        raise RuntimeError("exactly two admission rounds required")

    expected_target = _policy_target_identity(policy)
    expected_core_identity: tuple[tuple[object, ...], ...] | None = None
    selected_parent_cpus: tuple[int, ...] | None = None
    window_means: list[float] = []
    minimum_free = {node: float("inf") for node in policy.target.numa_nodes}

    for document in documents:
        if document.get("schema") != 1:
            raise RuntimeError("unsupported admission document schema")
        host_ips = document.get("host_ips")
        if not isinstance(host_ips, list) or policy.expected_host_ip not in host_ips:
            raise RuntimeError("admission document came from the wrong host")
        if _target_identity(document) != expected_target:
            raise RuntimeError("target identity changed between admission rounds")

        started_at = document.get("started_at")
        ended_at = document.get("ended_at")
        if not isinstance(started_at, (int, float)) or not isinstance(ended_at, (int, float)):
            raise RuntimeError("admission timing evidence is missing")
        if ended_at - started_at < policy.minimum_round_seconds:
            raise RuntimeError(
                "admission round shorter than "
                f"{policy.minimum_round_seconds:g} seconds"
            )

        windows = document.get("windows")
        if not isinstance(windows, list) or len(windows) != policy.windows_per_round:
            raise RuntimeError("each admission round requires six windows")

        _validate_training_identity(document.get("training_start"), policy)
        _validate_training_identity(document.get("training_end"), policy)

        cores = _selected_cores(document, policy)
        identity = _core_identity(cores)
        if expected_core_identity is None:
            expected_core_identity = identity
            selected_parent_cpus = tuple(min(row["siblings"]) for row in cores)
        elif identity != expected_core_identity:
            raise RuntimeError("candidate CPU identity changed between admission rounds")

        for raw_window in windows:
            window = _mapping(raw_window, "admission window")
            idle = _mapping(window.get("idle"), "CPU idle sample")
            mean = _conservative_mean(cores, idle)
            window_means.append(mean)
            if mean <= policy.minimum_idle_exclusive:
                raise RuntimeError("candidate CPU idle gate failed")

            system_meminfo = _mapping(
                window.get("system_meminfo"), "system memory sample"
            )
            if system_meminfo.get("MemAvailable", 0) < policy.minimum_system_available_kib:
                raise RuntimeError("system MemAvailable gate failed")

            numa_meminfo = _mapping(window.get("numa_meminfo"), "NUMA memory sample")
            for node in policy.target.numa_nodes:
                node_info = _mapping(numa_meminfo.get(str(node)), f"NUMA{node} memory sample")
                free_kib = node_info.get("MemFree", 0)
                if isinstance(free_kib, bool) or not isinstance(free_kib, int):
                    raise RuntimeError(f"NUMA{node} MemFree evidence is malformed")
                minimum_free[node] = min(minimum_free[node], free_kib / 1024**2)
                if free_kib < policy.minimum_node_free_kib:
                    raise RuntimeError(f"NUMA{node} MemFree gate failed")

            _validate_device_window(window, policy)

    if selected_parent_cpus is None:
        raise RuntimeError("candidate CPU identity is missing")
    return AdmissionResult(
        status="PASS",
        target=policy.target,
        numa_nodes=policy.target.numa_nodes,
        windows=len(window_means),
        selected_physical_cores=policy.expected_physical_cores,
        selected_parent_cpus=selected_parent_cpus,
        min_mean_idle=min(window_means),
        max_mean_idle=max(window_means),
        window_means=tuple(window_means),
        min_numa_free_gib=minimum_free,
    )


__all__ = (
    "AdmissionPolicy",
    "AdmissionResult",
    "validate_admission_documents",
)
