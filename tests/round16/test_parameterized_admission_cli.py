from __future__ import annotations

import unittest
from pathlib import Path
import subprocess
import sys

from tools.round16_admission import (
    claims_from_container_documents,
    policy_from_payload,
)


TRAINING_ID = "a" * 64


class ParameterizedAdmissionCliTests(unittest.TestCase):
    def test_direct_cli_entrypoint_resolves_sibling_round16_package(self) -> None:
        root = Path(__file__).resolve().parents[2]
        completed = subprocess.run(
            [sys.executable, "tools/round16_admission.py", "--help"],
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("precheck", completed.stdout)

    def test_policy_payload_carries_explicit_device_and_numa_identity(self) -> None:
        policy = policy_from_payload(
            {
                "schema": 1,
                "target": {
                    "physical_id": 10,
                    "card_id": 5,
                    "chip_id": 0,
                    "pci_bdf": "0000:89:00.0",
                    "numa_nodes": [2, 3, 4, 7],
                },
                "training_container_id": TRAINING_ID,
                "expected_host_ip": "110.123.0.3",
            }
        )

        self.assertEqual(policy.target.physical_id, 10)
        self.assertEqual(policy.target.numa_nodes, (2, 3, 4, 7))
        self.assertEqual(policy.training_container_id, TRAINING_ID)

    def test_container_claims_parse_single_or_comma_separated_visible_devices(self) -> None:
        claims = claims_from_container_documents(
            [
                {
                    "Id": "b" * 64,
                    "Name": "/owner-a",
                    "Config": {"Env": ["ASCEND_RT_VISIBLE_DEVICES=10,11"]},
                },
                {
                    "Id": "c" * 64,
                    "Name": "/owner-b",
                    "Config": {"Env": ["ASCEND_VISIBLE_DEVICES=9"]},
                },
            ]
        )

        self.assertEqual(
            claims, {9: "/owner-b", 10: "/owner-a", 11: "/owner-a"}
        )

    def test_duplicate_claim_and_malformed_visible_device_fail_closed(self) -> None:
        first = {
            "Id": "b" * 64,
            "Name": "/first",
            "Config": {"Env": ["ASCEND_RT_VISIBLE_DEVICES=10"]},
        }
        second = {
            "Id": "c" * 64,
            "Name": "/second",
            "Config": {"Env": ["ASCEND_RT_VISIBLE_DEVICES=10"]},
        }
        with self.assertRaisesRegex(RuntimeError, "multiple containers claim"):
            claims_from_container_documents([first, second])

        malformed = {
            "Id": "d" * 64,
            "Name": "/bad",
            "Config": {"Env": ["ASCEND_RT_VISIBLE_DEVICES=10,bad"]},
        }
        with self.assertRaisesRegex(RuntimeError, "malformed ASCEND"):
            claims_from_container_documents([malformed])


if __name__ == "__main__":
    unittest.main()
