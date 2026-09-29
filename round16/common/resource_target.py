"""Pure, fail-closed selection of an explicit Round16 NPU/NUMA target.

This module performs no sampling, device initialization, Docker mutation, or
process management.  It only normalizes already captured ``npu-smi`` and
container-claim evidence, then checks explicitly configured targets.  The
full two-round CPU/NUMA/HBM admission remains a separate mandatory step.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Mapping, Sequence


_PCI_BDF = re.compile(r"^0000:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-7]$")
_CARD_ROW = re.compile(r"^\|\s*(\d+)\s+Ascend\S*\s*\|")
_DEVICE_ROW = re.compile(
    r"^\|\s*(\d+)\s+(\d+)\s*\|\s*(0000:[0-9A-Fa-f:.]+)\s*\|"
    r".*?(\d+)\s*/\s*(\d+)\s*\|\s*$"
)
_PROCESS_ROW = re.compile(
    r"^\|\s*(\d+)\s+(\d+)\s*\|\s*(\d+)\s*\|\s*([^|]+?)\s*\|"
    r"\s*(\d+)\s*\|"
)
_CLAIM_ROW = re.compile(
    r"^([^|]+)\|ASCEND_RT_VISIBLE_DEVICES=(\d+)\s*$"
)


@dataclass(frozen=True)
class NpuProcess:
    pid: int
    name: str
    memory_mib: int


@dataclass(frozen=True)
class NpuDeviceSnapshot:
    physical_id: int
    card_id: int
    chip_id: int
    pci_bdf: str
    hbm_used_mib: int
    hbm_total_mib: int
    processes: tuple[NpuProcess, ...] = ()


@dataclass(frozen=True)
class NpuTarget:
    physical_id: int
    card_id: int
    chip_id: int
    pci_bdf: str
    numa_nodes: tuple[int, int, int, int]

    def __post_init__(self) -> None:
        integer_fields = (self.physical_id, self.card_id, self.chip_id)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in integer_fields):
            raise TypeError("device identities must be integers")
        if not 0 <= self.physical_id <= 15:
            raise ValueError("physical device must be in the frozen 0..15 inventory")
        if not 0 <= self.card_id <= 7 or self.chip_id not in (0, 1):
            raise ValueError("card/chip identity is outside the frozen 8x2 topology")
        if type(self.pci_bdf) is not str or _PCI_BDF.fullmatch(self.pci_bdf) is None:
            raise ValueError("PCI BDF must use the full 0000:BB:DD.F form")
        normalized_nodes = tuple(self.numa_nodes)
        if (
            len(normalized_nodes) != 4
            or len(set(normalized_nodes)) != 4
            or any(
                isinstance(node, bool) or not isinstance(node, int) or node < 0
                for node in normalized_nodes
            )
        ):
            raise ValueError("target requires four distinct NUMA node identities")
        object.__setattr__(self, "pci_bdf", self.pci_bdf.upper())
        object.__setattr__(self, "numa_nodes", normalized_nodes)


@dataclass(frozen=True)
class TargetDecision:
    ready: bool
    target: NpuTarget | None
    rejections: Mapping[int, tuple[str, ...]]


def parse_npu_smi(raw: str) -> dict[int, NpuDeviceSnapshot]:
    """Parse physical inventory and the card/chip process table together."""

    if type(raw) is not str:
        raise TypeError("npu-smi evidence must be text")
    current_card: int | None = None
    in_process_table = False
    devices: dict[int, NpuDeviceSnapshot] = {}
    processes: dict[tuple[int, int], list[NpuProcess]] = {}

    for line in raw.splitlines():
        if "Process id" in line and "NPU" in line and "Chip" in line:
            in_process_table = True
            current_card = None
            continue
        if not in_process_table:
            if match := _CARD_ROW.match(line):
                current_card = int(match.group(1))
                continue
            if match := _DEVICE_ROW.match(line):
                if current_card is None:
                    raise ValueError("device row has no preceding card identity")
                chip_id = int(match.group(1))
                physical_id = int(match.group(2))
                if physical_id in devices:
                    raise ValueError(f"duplicate physical device row {physical_id}")
                devices[physical_id] = NpuDeviceSnapshot(
                    physical_id=physical_id,
                    card_id=current_card,
                    chip_id=chip_id,
                    pci_bdf=match.group(3).upper(),
                    hbm_used_mib=int(match.group(4)),
                    hbm_total_mib=int(match.group(5)),
                )
                current_card = None
                continue
        elif match := _PROCESS_ROW.match(line):
            key = (int(match.group(1)), int(match.group(2)))
            processes.setdefault(key, []).append(
                NpuProcess(
                    pid=int(match.group(3)),
                    name=match.group(4).strip(),
                    memory_mib=int(match.group(5)),
                )
            )

    if not devices:
        raise ValueError("npu-smi evidence contains no physical devices")
    result: dict[int, NpuDeviceSnapshot] = {}
    for physical_id in sorted(devices):
        device = devices[physical_id]
        result[physical_id] = NpuDeviceSnapshot(
            physical_id=device.physical_id,
            card_id=device.card_id,
            chip_id=device.chip_id,
            pci_bdf=device.pci_bdf,
            hbm_used_mib=device.hbm_used_mib,
            hbm_total_mib=device.hbm_total_mib,
            processes=tuple(processes.get((device.card_id, device.chip_id), ())),
        )
    unknown_process_keys = set(processes).difference(
        (device.card_id, device.chip_id) for device in result.values()
    )
    if unknown_process_keys:
        raise ValueError(f"process table references unknown card/chip: {sorted(unknown_process_keys)}")
    return result


def parse_device_claims(raw: str) -> dict[int, str]:
    """Parse single-device container claims and reject ambiguous evidence."""

    if type(raw) is not str:
        raise TypeError("device claim evidence must be text")
    claims: dict[int, str] = {}
    for line in raw.splitlines():
        if not line.strip():
            continue
        match = _CLAIM_ROW.fullmatch(line.strip())
        if match is None:
            raise ValueError(f"malformed device claim row: {line!r}")
        owner = match.group(1)
        physical_id = int(match.group(2))
        if not 0 <= physical_id <= 15:
            raise ValueError("physical device claim is outside the frozen 0..15 inventory")
        if physical_id in claims:
            raise ValueError(f"duplicate device claim for physical device {physical_id}")
        claims[physical_id] = owner
    return claims


def select_safe_target(
    devices: Mapping[int, NpuDeviceSnapshot],
    *,
    claims: Mapping[int, str],
    allowed_training_pids: set[int],
    candidates: Sequence[NpuTarget],
    expected_total_mib: int = 65536,
    training_process_limit_mib: int = 15360,
) -> TargetDecision:
    """Select only an explicit identity with no external owner or process.

    This is the ownership/identity gate before the two 60-second admission
    rounds.  It deliberately does not infer a NUMA profile from PCI locality.
    Every candidate carries an already reviewed four-node profile.
    """

    if not candidates:
        raise ValueError("at least one explicit target candidate is required")
    rejections: dict[int, tuple[str, ...]] = {}
    for target in candidates:
        reasons: list[str] = []
        device = devices.get(target.physical_id)
        if device is None:
            reasons.append("DEVICE_NOT_PRESENT")
        else:
            if target.physical_id in claims:
                reasons.append("EXTERNAL_CONTAINER_CLAIM")
            if (
                device.card_id != target.card_id
                or device.chip_id != target.chip_id
                or device.pci_bdf.upper() != target.pci_bdf.upper()
            ):
                reasons.append("CARD_CHIP_OR_BUS_IDENTITY_MISMATCH")
            if device.hbm_total_mib != expected_total_mib:
                reasons.append("HBM_TOTAL_IDENTITY_MISMATCH")
            unknown = [
                process
                for process in device.processes
                if process.pid not in allowed_training_pids
            ]
            if unknown:
                reasons.append("UNKNOWN_DEVICE_PROCESS")
            if any(
                process.pid in allowed_training_pids
                and process.memory_mib > training_process_limit_mib
                for process in device.processes
            ):
                reasons.append("TRAINING_HBM_ALLOWANCE_EXCEEDED")
        if not reasons:
            return TargetDecision(True, target, {})
        rejections[target.physical_id] = tuple(reasons)
    return TargetDecision(False, None, rejections)


__all__ = (
    "NpuDeviceSnapshot",
    "NpuProcess",
    "NpuTarget",
    "TargetDecision",
    "parse_device_claims",
    "parse_npu_smi",
    "select_safe_target",
)
