from __future__ import annotations

import copy
import unittest

from round16.common.parameterized_admission import (
    AdmissionPolicy,
    validate_admission_documents,
)
from round16.common.resource_target import NpuTarget


TRAINING_ID = "0" * 64


def _policy() -> AdmissionPolicy:
    return AdmissionPolicy(
        target=NpuTarget(10, 5, 0, "0000:89:00.0", (2, 3, 4, 7)),
        training_container_id=TRAINING_ID,
        expected_host_ip="110.123.0.3",
    )


def _document(*, started_at: float = 1.0) -> dict[str, object]:
    cores: list[dict[str, object]] = []
    idle: dict[str, float] = {}
    logical_cpu = 0
    for node in (2, 3, 4, 7):
        for core_id in range(40):
            siblings = [logical_cpu, logical_cpu + 320]
            logical_cpu += 1
            cores.append(
                {
                    "node": node,
                    "socket": node // 2,
                    "core": core_id,
                    "siblings": siblings,
                    "excluded_explicit_container_affinity": False,
                }
            )
            idle.update({str(cpu): 0.99 for cpu in siblings})

    training = {
        "id": TRAINING_ID,
        "running": True,
        "pid": 123,
        "cgroup": f"0::/docker/{TRAINING_ID}\n",
    }
    window = {
        "idle": idle,
        "system_meminfo": {"MemAvailable": 705 * 1024**2},
        "numa_meminfo": {
            str(node): {"MemFree": 49 * 1024**2}
            for node in (2, 3, 4, 7)
        },
        "target_device": {
            "physical_id": 10,
            "card_id": 5,
            "chip_id": 0,
            "pci_bdf": "0000:89:00.0",
            "hbm_used_mib": 6124,
            "hbm_total_mib": 65536,
            "processes": [
                {
                    "pid": 456,
                    "name": "python3.10",
                    "memory_mib": 6124,
                    "cgroup": f"0::/docker/{TRAINING_ID}\n",
                }
            ],
        },
        "device_claims": {},
    }
    return {
        "schema": 1,
        "started_at": started_at,
        "ended_at": started_at + 60.0,
        "host_ips": ["110.123.0.3"],
        "target": {
            "physical_id": 10,
            "card_id": 5,
            "chip_id": 0,
            "pci_bdf": "0000:89:00.0",
            "numa_nodes": [2, 3, 4, 7],
        },
        "cores": cores,
        "training_start": training,
        "training_end": training,
        "windows": [copy.deepcopy(window) for _ in range(6)],
    }


def _ten_second_policy() -> AdmissionPolicy:
    return AdmissionPolicy(
        target=NpuTarget(10, 5, 0, "0000:89:00.0", (2, 3, 4, 7)),
        training_container_id=TRAINING_ID,
        expected_host_ip="110.123.0.3",
        windows_per_round=1,
        minimum_round_seconds=10.0,
    )


def _ten_second_document(*, started_at: float = 1.0) -> dict[str, object]:
    document = _document(started_at=started_at)
    document["ended_at"] = started_at + 10.0
    document["windows"] = document["windows"][:1]
    return document


class ParameterizedAdmissionTests(unittest.TestCase):
    def test_two_ten_second_rounds_pass_when_policy_requests_them(self) -> None:
        result = validate_admission_documents(
            [
                _ten_second_document(started_at=1),
                _ten_second_document(started_at=21),
            ],
            _ten_second_policy(),
        )

        self.assertEqual(result.status, "PASS")
        self.assertEqual(result.windows, 2)

    def test_two_complete_rounds_pass_for_explicit_target_and_numa_profile(self) -> None:
        result = validate_admission_documents(
            [_document(started_at=1), _document(started_at=101)], _policy()
        )

        self.assertEqual(result.status, "PASS")
        self.assertEqual(result.target.physical_id, 10)
        self.assertEqual(result.numa_nodes, (2, 3, 4, 7))
        self.assertEqual(result.windows, 12)
        self.assertEqual(result.selected_physical_cores, 160)
        self.assertGreater(result.min_mean_idle, 0.80)

    def test_rejects_target_or_cpu_identity_drift_between_rounds(self) -> None:
        second = _document(started_at=101)
        second["target"]["physical_id"] = 12
        with self.assertRaisesRegex(RuntimeError, "target identity"):
            validate_admission_documents([_document(), second], _policy())

        second = _document(started_at=101)
        second["cores"][0]["siblings"] = [999, 1000]
        with self.assertRaisesRegex(RuntimeError, "candidate CPU identity"):
            validate_admission_documents([_document(), second], _policy())

    def test_rejects_external_claim_even_when_only_training_process_is_visible(self) -> None:
        document = _document()
        document["windows"][2]["device_claims"] = {"10": "/external-owner"}

        with self.assertRaisesRegex(RuntimeError, "external container claim"):
            validate_admission_documents(
                [document, _document(started_at=101)], _policy()
            )

    def test_rejects_unknown_process_or_training_hbm_above_allowance(self) -> None:
        document = _document()
        document["windows"][0]["target_device"]["processes"][0]["cgroup"] = (
            "0::/docker/" + "1" * 64 + "\n"
        )
        with self.assertRaisesRegex(RuntimeError, "unknown target process"):
            validate_admission_documents(
                [document, _document(started_at=101)], _policy()
            )

        document = _document()
        document["windows"][0]["target_device"]["processes"][0][
            "memory_mib"
        ] = 15361
        with self.assertRaisesRegex(RuntimeError, "training process exceeds"):
            validate_admission_documents(
                [document, _document(started_at=101)], _policy()
            )

    def test_rejects_low_idle_node_free_or_system_available(self) -> None:
        document = _document()
        document["windows"][0]["idle"] = {
            cpu: 0.80 for cpu in document["windows"][0]["idle"]
        }
        with self.assertRaisesRegex(RuntimeError, "CPU idle"):
            validate_admission_documents(
                [document, _document(started_at=101)], _policy()
            )

        document = _document()
        document["windows"][0]["numa_meminfo"]["4"]["MemFree"] = 48 * 1024**2 - 1
        with self.assertRaisesRegex(RuntimeError, "NUMA4"):
            validate_admission_documents(
                [document, _document(started_at=101)], _policy()
            )

        document = _document()
        document["windows"][0]["system_meminfo"]["MemAvailable"] = 704 * 1024**2 - 1
        with self.assertRaisesRegex(RuntimeError, "system MemAvailable"):
            validate_admission_documents(
                [document, _document(started_at=101)], _policy()
            )

    def test_rejects_wrong_hbm_target_or_short_sampling(self) -> None:
        document = _document()
        document["windows"][0]["target_device"]["pci_bdf"] = "0000:85:00.0"
        with self.assertRaisesRegex(RuntimeError, "target device identity"):
            validate_admission_documents(
                [document, _document(started_at=101)], _policy()
            )

        document = _document()
        document["windows"][0]["target_device"]["hbm_total_mib"] = 32768
        with self.assertRaisesRegex(RuntimeError, "HBM total"):
            validate_admission_documents(
                [document, _document(started_at=101)], _policy()
            )

        document = _document()
        document["ended_at"] = document["started_at"] + 59.9
        with self.assertRaisesRegex(RuntimeError, "shorter than 60"):
            validate_admission_documents(
                [document, _document(started_at=101)], _policy()
            )

    def test_policy_requires_full_training_identity_and_frozen_160_core_set(self) -> None:
        with self.assertRaisesRegex(ValueError, "full Docker ID"):
            AdmissionPolicy(
                target=NpuTarget(10, 5, 0, "0000:89:00.0", (2, 3, 4, 7)),
                training_container_id="short",
                expected_host_ip="110.123.0.3",
            )

        document = _document()
        document["cores"].pop()
        with self.assertRaisesRegex(RuntimeError, "160-core"):
            validate_admission_documents(
                [document, _document(started_at=101)], _policy()
            )


if __name__ == "__main__":
    unittest.main()
