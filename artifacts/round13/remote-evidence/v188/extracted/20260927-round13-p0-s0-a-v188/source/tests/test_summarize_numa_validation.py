import json
from pathlib import Path
import tempfile
import unittest

from migration_bridge.round9.tools import summarize_numa_validation as api


class SummarizeNumaValidationTests(unittest.TestCase):
    def sample(self, free_gib, *, anon_pages=0, usage=0, failcnt="0", events=None):
        kib = 1024 ** 2
        node = {
            "MemFree": int(free_gib * kib),
            "Inactive(file)": 100 * kib,
            "Dirty": 0,
            "Writeback": 0,
            "Shmem": 0,
            "Mapped": 10 * kib,
        }
        cgroup = {
            "memory.failcnt": failcnt,
            "memory.events": events or "low 0\nhigh 0\noom 0\noom_kill 0",
            "memory.usage_in_bytes": str(usage),
            "memory.max_usage_in_bytes": str(usage),
            "memory.numa_stat": (
                "total=0 N2=0\nfile=0 N2=0\n"
                f"anon={anon_pages} N2={anon_pages}\n"
            ),
        }
        return {
            "numa_meminfo_kib": {"2": node},
            "owned_cgroup": cgroup,
            "hbm_used_mib": 12000,
            "estimated_nonowned_board_mib": 2000,
        }

    def test_summarizes_minimum_headroom_and_owned_peaks(self):
        rows = [
            {"time": 1.0, "action": "STARTING"},
            self.sample(60, anon_pages=2 * 1024 ** 2, usage=8 * 1024 ** 3),
            self.sample(7, anon_pages=3 * 1024 ** 2, usage=9 * 1024 ** 3),
        ]
        result = api.summarize(rows, [2], driver_status="PASS_TWO_DRIVERS")
        self.assertEqual(result["status"], "PASS_MONITORED_MXFP4_NUMA_VALIDATION")
        self.assertEqual(result["samples"], 3)
        self.assertEqual(result["cgroup_samples"], 2)
        self.assertEqual(result["nodes"]["2"]["minimum_memfree_gib"], 7.0)
        self.assertEqual(result["nodes"]["2"]["minimum_conservative_available_gib"], 89.0)
        self.assertEqual(result["nodes"]["2"]["peak_owned_anon_gib"], 12.0)
        self.assertEqual(result["peak_owned_cgroup_usage_gib"], 9.0)
        self.assertEqual(result["peak_owned_hbm_gib"], 10000 / 1024)

    def test_pressure_or_incomplete_driver_cannot_pass(self):
        pressure = self.sample(60, failcnt="1")
        result = api.summarize([pressure], [2], driver_status="PASS_TWO_DRIVERS")
        self.assertEqual(result["status"], "FAIL_MONITORED_MXFP4_NUMA_VALIDATION")
        self.assertTrue(any("allocation failure" in item for item in result["failures"]))
        result = api.summarize([self.sample(60)], [2], driver_status="NOT_RUN")
        self.assertEqual(result["status"], "INCOMPLETE_MONITORED_MXFP4_NUMA_VALIDATION")

    def test_cli_writes_deterministic_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            monitor = root / "monitor.jsonl"
            monitor.write_text(json.dumps(self.sample(60)) + "\n")
            driver = root / "driver.json"
            driver.write_text(json.dumps({"status": "PASS_TWO_DRIVERS"}))
            output = root / "summary.json"
            self.assertEqual(api.main([
                "--monitor", str(monitor), "--driver", str(driver),
                "--nodes", "2", "--output", str(output),
            ]), 0)
            saved = json.loads(output.read_text())
            self.assertEqual(saved["status"], "PASS_MONITORED_MXFP4_NUMA_VALIDATION")
            self.assertTrue(output.read_text().endswith("\n"))


if __name__ == "__main__":
    unittest.main()
