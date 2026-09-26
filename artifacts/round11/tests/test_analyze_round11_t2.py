import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from artifacts.round11.analyze_round11_t2 import compose_t2


CAPTURE = "c837641468b9db93c422a3610af7a7ecc1f14d4bbb359841d69ac0dfd655dc6e"
BANK = "e9d6b27f352d501a9ed3737ec5d3ab140b9d1f5bda22d2e36ef9ef0bcc45b08b"
OUTPUT = "f5e189843493b9f42fcf724d9487ec5e1b9c630a6b0e780079cbc2bf6053f7c4"
BASE_NATIVE = "614ae60f70ff70d1e47d7fd32e857f887f874b6248eb332172c8a637cb471a6d"
ATTR_NATIVE = "dff773ecabb3c7419afd8e3586e3c09934ebd9f23c7d9604eca5fa1218b39183"


def summary(value):
    return {
        "count": 3000,
        "median_us": value,
        "p05_us": value * 0.9,
        "p95_us": value * 1.1,
        "min_us": value * 0.8,
        "max_us": value * 1.2,
        "mean_us": value,
    }


def result(role):
    attributed = role == "attribution"
    counters = role == "counters"
    total = 1500 if role == "baseline" else 1510 if counters else 1520
    document = {
        "schema": 1,
        "status": "PASS_ROUND11_T2_DEEP_ATTRIBUTION",
        "scope": "ROUND11_T2_REAL_LAYER0_C128_MXFP4_DEEP_ATTRIBUTION",
        "iterations": 3000,
        "warmups": 5,
        "tokens": 3,
        "numa_nodes": [2, 3, 4, 7],
        "cpu_workers": 128,
        "threadpool_count": 4,
        "capture_sha256": CAPTURE,
        "bank_sha256": BANK,
        "native_sha256": ATTR_NATIVE if attributed else BASE_NATIVE,
        "binary_role": (
            "ATTRIBUTION_ONLY_TASK_QUEUE_TIMESTAMPS"
            if attributed
            else "FROZEN_HFINAL_NATIVE"
        ),
        "output_sha256": OUTPUT,
        "repeat_hash_exact": True,
        "parent_and_exclusive_us": {
            "total_parent": summary(total),
            "submit_parent": summary(160),
            "pinned_gather_and_metadata": summary(100),
            "enqueue_and_task_construction_exclusive": summary(60),
            "sync_parent": summary(total - 160),
            "exposed_queue_and_compute_wait_exclusive": summary(total - 200),
            "result_publication": summary(40),
        },
        "native_phases": {
            "tp_partitions": {
                str(i): {
                    "calls": 9015,
                    "quant_us_per_row": 10 + i,
                    "gateup_activation_us_per_row": 200 + i,
                    "down_weighted_sum_us_per_row": 180 + i,
                }
                for i in range(4)
            },
            "critical_tp_compute_us_per_group": 1197,
            "fastest_tp_compute_us_per_group": 1170,
            "tp_partition_imbalance_ratio": 1197 / 1170,
            "merge": {"calls": 3005, "merge_publication_us_per_group": 20},
            "native_compute_plus_merge_us_per_group": 1217,
        },
        "queue_wait_separation": (
            {
                "calls": 3005,
                "queue_admission_wait_us_per_group": 30,
                "task_execute_us_per_group": 1230,
            }
            if attributed
            else "BOUNDED_RESIDUAL_ONLY_WITH_FROZEN_BINARY"
        ),
        "thread_accounting": {
            "elapsed_ns": 4_500_000_000,
            "clock_ticks_per_second": 100,
            "rows": [
                {
                    "tid": i,
                    "name": "worker",
                    "sched_runtime_ns": 1_000_000 + i,
                    "sched_wait_ns": 10_000,
                    "sched_timeslices": 20,
                    "cpu_ticks": 1,
                }
                for i in range(1, 129)
            ],
        },
        "hardware_counters": (
            {
                "status": "PASS_ALL_EXISTING_THREADS",
                "summary": {
                    "scaled_totals": {
                        "cycles": 3_000_000,
                        "instructions": 6_000_000,
                        "cache_references": 600_000,
                        "cache_misses": 60_000,
                        "context_switches": 300,
                        "cpu_migrations": 3,
                    },
                    "ipc": 2.0,
                    "generic_cache_miss_ratio": 0.1,
                },
                "llc_misses": "UNKNOWN_GENERIC_CACHE_EVENT_IS_NOT_LLC_SPECIFIC",
                "memory_controller_bandwidth": "UNKNOWN_NO_SUPPORTED_CONTROLLER_EVENT_MAPPING",
                "remote_numa_accesses": "UNKNOWN_NO_SUPPORTED_UNCORE_OR_NUMA_ACCESS_EVENT_MAPPING",
            }
            if counters
            else {"status": "DISABLED_FOR_TIMING_ISOLATION"}
        ),
    }
    return document


def write_run(root, role, version):
    path = root / role
    path.mkdir()
    raw = result(role)
    result_path = path / "result.json"
    result_path.write_text(json.dumps(raw))
    digest = hashlib.sha256(result_path.read_bytes()).hexdigest()
    controller = {
        "status": "PASS_ROUND11_T2_COMPONENT_CONTROLLER",
        "role": role,
        "version": version,
        "result_sha256": digest,
        "guard_exit_ack": {"action": "OWNED_EXITED_ACK"},
        "owned_container_removed": True,
        "physical12_after": "EMPTY",
        "oom_killed": False,
    }
    (path / "controller.json").write_text(json.dumps(controller))
    (path / "post-cleanup-proc-mem.txt").write_text("No process in device.\n")
    return path


class AnalyzeRound11T2Tests(unittest.TestCase):
    def test_composes_exclusive_hierarchy_without_summing_parallel_tp(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            runs = {
                "baseline": write_run(root, "baseline", 113),
                "counters": write_run(root, "counters", 114),
                "attribution": write_run(root, "attribution", 115),
            }
            report = compose_t2(runs)
        self.assertEqual(report["status"], "PASS_ROUND11_T2_DEEP_ATTRIBUTION")
        decomposition = report["attribution_decomposition_us_per_group"]
        self.assertEqual(decomposition["total_parent"], 1520)
        self.assertEqual(decomposition["native_parallel_critical_compute_plus_merge"], 1217)
        self.assertEqual(decomposition["queue_admission_wait"], 30)
        self.assertEqual(decomposition["task_execute_residual_after_native"], 13)
        self.assertEqual(decomposition["sync_bridge_residual"], 60)
        self.assertAlmostEqual(report["instrumentation_overhead_ratio"], 1520 / 1500)
        self.assertEqual(report["hardware_counters"]["ipc"], 2.0)
        self.assertEqual(report["hardware_counters"]["cycles_per_group"], 1000)

    def test_rejects_controller_hash_drift(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            runs = {
                "baseline": write_run(root, "baseline", 113),
                "counters": write_run(root, "counters", 114),
                "attribution": write_run(root, "attribution", 115),
            }
            controller_path = runs["baseline"] / "controller.json"
            controller = json.loads(controller_path.read_text())
            controller["result_sha256"] = "0" * 64
            controller_path.write_text(json.dumps(controller))
            with self.assertRaisesRegex(ValueError, "result hash"):
                compose_t2(runs)

    def test_preserves_unavailable_hardware_counter_boundaries(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            runs = {
                "baseline": write_run(root, "baseline", 113),
                "counters": write_run(root, "counters", 114),
                "attribution": write_run(root, "attribution", 115),
            }
            counter_path = runs["counters"] / "result.json"
            counter = json.loads(counter_path.read_text())
            counter["hardware_counters"] = {
                "status": "PARTIAL_OR_UNAVAILABLE",
                "open_errors": [{"errno": 13, "error": "Permission denied"}],
                "llc_misses": "UNKNOWN_GENERIC_CACHE_EVENT_IS_NOT_LLC_SPECIFIC",
                "memory_controller_bandwidth": "UNKNOWN_NO_SUPPORTED_CONTROLLER_EVENT_MAPPING",
                "remote_numa_accesses": "UNKNOWN_NO_SUPPORTED_UNCORE_OR_NUMA_ACCESS_EVENT_MAPPING",
            }
            counter_path.write_text(json.dumps(counter))
            controller_path = runs["counters"] / "controller.json"
            controller = json.loads(controller_path.read_text())
            controller["result_sha256"] = hashlib.sha256(counter_path.read_bytes()).hexdigest()
            controller_path.write_text(json.dumps(controller))
            report = compose_t2(runs)
        self.assertEqual(report["hardware_counters"]["status"], "PARTIAL_OR_UNAVAILABLE")
        self.assertEqual(report["hardware_counters"]["memory_bandwidth"], "UNKNOWN")


if __name__ == "__main__":
    unittest.main()
