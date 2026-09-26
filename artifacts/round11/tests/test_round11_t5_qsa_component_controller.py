import time
import unittest
import tempfile
import hashlib
import json
from pathlib import Path

from artifacts.round11.round11_t5_qsa_component_controller import (
    TRAINING_ID,
    build_container_command,
    stage_guard_sources,
    validate_admission,
    validate_candidate_manifest,
)


class Round11T5QSAComponentControllerTests(unittest.TestCase):
    def test_guard_staging_includes_numa_policy_dependency(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            monitor = root / "monitor.py"
            numa = root / "numa.py"
            source = root / "out"
            monitor.write_text("monitor\n")
            numa.write_text("numa\n")
            stage_guard_sources(monitor, numa, source)
            self.assertEqual((source / "round11_monitor_owned.py").read_text(), "monitor\n")
            self.assertEqual((source / "numa_policy.py").read_text(), "numa\n")

    def test_container_command_freezes_qsa_candidate_identity(self):
        command = build_container_command(
            version=144,
            output=Path("/evidence"),
            cpus=[160, 162, 560, 562],
            candidate=Path("/mnt/data1/kt-round9/round11-t5-qsa-component-v144"),
        )
        self.assertEqual(
            command[command.index("--name") + 1],
            "kt-round11-t5-qsa-candidate-v144",
        )
        self.assertEqual(command[command.index("--cpuset-mems") + 1], "2,3,4,7")
        self.assertEqual(command[command.index("--cpus") + 1], "136")
        self.assertIn("/dev/davinci12", command)
        self.assertTrue(any(item.endswith(":/candidate:ro") for item in command))
        self.assertTrue(any(item.endswith(
            ":/models/de4b8e4d43b917e7706784d8bb445c9af86a3540:ro"
        ) for item in command))
        self.assertIn("--iterations 9", command[-1])

    def test_admission_requires_fresh_exact_shared_identity(self):
        now = time.time()
        document = {
            "status": "PASS",
            "ended_at": now,
            "windows": 12,
            "training_id": TRAINING_ID,
            "selected_physical_cores": 160,
            "min_mean_idle": 0.99,
            "min_numa_free_gib": {"2": 60, "3": 60, "4": 60, "7": 60},
            "cpus": list(range(160)),
        }
        self.assertEqual(len(validate_admission(document, now=now)), 160)
        document["ended_at"] = now - 901
        with self.assertRaisesRegex(ValueError, "stale"):
            validate_admission(document, now=now)

    def test_candidate_manifest_rejects_remote_source_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "candidate.py"
            source.write_text("frozen\n")
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            (root / "source-sha256.json").write_text(json.dumps({
                "candidate.py": digest,
            }))
            self.assertEqual(validate_candidate_manifest(root), {"candidate.py": digest})
            source.write_text("drift\n")
            with self.assertRaisesRegex(ValueError, "manifest"):
                validate_candidate_manifest(root)


if __name__ == "__main__":
    unittest.main()
