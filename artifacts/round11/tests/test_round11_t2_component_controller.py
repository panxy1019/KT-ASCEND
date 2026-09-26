import json
import tempfile
import time
import unittest
from pathlib import Path

from artifacts.round11.round11_t2_component_controller import (
    ROLE_CONTRACT,
    build_container_command,
    validate_admission,
)


class Round11T2ComponentControllerTests(unittest.TestCase):
    def test_role_contract_freezes_versions_and_instrumentation(self):
        self.assertEqual(ROLE_CONTRACT["baseline"], (113, "OFF", "baseline"))
        self.assertEqual(ROLE_CONTRACT["counters"], (114, "ON", "baseline"))
        self.assertEqual(ROLE_CONTRACT["attribution"], (115, "OFF", "attribution"))

    def test_container_command_has_c128_numa_and_bounded_identity(self):
        command = build_container_command(
            role="baseline",
            version=113,
            output=Path("/evidence"),
            cpus=[160, 162, 560, 562],
            runtime=Path("/mnt/data1/kt-round9/candidate-runtime-python"),
        )
        self.assertEqual(command[command.index("--name") + 1], "kt-round11-t2-baseline-v113")
        self.assertEqual(command[command.index("--cpuset-mems") + 1], "2,3,4,7")
        self.assertEqual(command[command.index("--cpus") + 1], "136")
        self.assertIn("/dev/davinci12", command)
        self.assertIn("PYTHONPATH=/round9/candidate-runtime-python", command)
        self.assertIn("--hardware-counters OFF", command[-1])

    def test_admission_requires_fresh_exact_identity(self):
        now = time.time()
        document = {
            "status": "PASS",
            "ended_at": now,
            "windows": 12,
            "training_id": "0419bcfee46f317d6833e038c28e81cf3d3ecf47323a423062bac2764e0ab2ef",
            "selected_physical_cores": 160,
            "min_mean_idle": 0.99,
            "min_numa_free_gib": {"2": 60, "3": 60, "4": 60, "7": 60},
            "cpus": list(range(160)),
        }
        self.assertEqual(len(validate_admission(document, now=now)), 160)
        document["min_numa_free_gib"]["4"] = 47.9
        with self.assertRaisesRegex(ValueError, "NUMA"):
            validate_admission(document, now=now)


if __name__ == "__main__":
    unittest.main()
