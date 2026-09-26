import importlib.util
from pathlib import Path
import time
import unittest


PATH = Path(__file__).resolve().parents[1] / "round11_t2_attribution_probe.py"
SPEC = importlib.util.spec_from_file_location("round11_t2_attribution_probe", PATH)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


class Round11T2AttributionTests(unittest.TestCase):
    def test_phase_parser_uses_parallel_tp_critical_path_and_separate_merge(self):
        lines = []
        for tp, values in enumerate(
            ((10, 20, 30), (11, 25, 31), (9, 22, 29), (10, 21, 30))
        ):
            lines.append(
                f"[KT_PHASE tp{tp}] n=3015 avg/layer-call: "
                f"quant={values[0]}.0us gateup={values[1]}.0us down={values[2]}.0us"
            )
        lines.append("[KT_PHASE merge] n=1005 avg/layer-call: merge=40.0us (qlen=3)")
        result = probe.parse_phase_log(
            "\n".join(lines),
            expected_tp_calls=3015,
            expected_merge_calls=1005,
            qlen=3,
        )
        self.assertEqual(result["critical_tp_compute_us_per_group"], 3 * (11 + 25 + 31))
        self.assertEqual(result["native_compute_plus_merge_us_per_group"], 241)
        self.assertGreater(result["tp_partition_imbalance_ratio"], 1)

    def test_phase_parser_rejects_missing_partition_or_call_drift(self):
        raw = "\n".join(
            f"[KT_PHASE tp{tp}] n=30 avg/layer-call: quant=1.0us gateup=2.0us down=3.0us"
            for tp in range(3)
        ) + "\n[KT_PHASE merge] n=10 avg/layer-call: merge=4.0us"
        with self.assertRaisesRegex(RuntimeError, "missing TP"):
            probe.parse_phase_log(raw, expected_tp_calls=30, expected_merge_calls=10, qlen=3)

    def test_percentile_and_summary_are_deterministic(self):
        result = probe.summarize_us([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual(result["median_us"], 3.0)
        self.assertAlmostEqual(result["p05_us"], 1.2)
        self.assertAlmostEqual(result["p95_us"], 4.8)

    def test_task_queue_parser_separates_wait_and_execute(self):
        raw = (
            "noise\n"
            "[KT_TASK_QUEUE] n=3005 avg/task: "
            "queue_admission_wait=8.5us task_execute=1301.2us\n"
        )
        self.assertEqual(
            probe.parse_task_queue_log(raw, expected_calls=3005),
            {
                "calls": 3005,
                "queue_admission_wait_us_per_group": 8.5,
                "task_execute_us_per_group": 1301.2,
            },
        )

    def test_task_queue_parser_rejects_call_drift(self):
        raw = (
            "[KT_TASK_QUEUE] n=3 avg/task: "
            "queue_admission_wait=1.0us task_execute=2.0us"
        )
        with self.assertRaisesRegex(RuntimeError, "call count"):
            probe.parse_task_queue_log(raw, expected_calls=3005)

    def test_admission_requires_fresh_twelve_window_identity(self):
        document = {
            "status": "PASS",
            "ended_at": time.time() - 10,
            "windows": 12,
            "training_id": probe.TRAINING_ID,
            "selected_physical_cores": 160,
            "min_mean_idle": 0.9,
        }
        self.assertEqual(probe.validate_admission(document)["windows"], 12)
        document["ended_at"] = time.time() - 901
        with self.assertRaisesRegex(ValueError, "stale"):
            probe.validate_admission(document)

    def test_perf_counter_summary_scales_multiplexed_counts(self):
        rows = [
            {"tid": 1, "event": "cycles", "raw": 50, "time_enabled": 100, "time_running": 50},
            {"tid": 2, "event": "cycles", "raw": 25, "time_enabled": 100, "time_running": 50},
            {"tid": 1, "event": "instructions", "raw": 100, "time_enabled": 100, "time_running": 100},
            {"tid": 2, "event": "instructions", "raw": 50, "time_enabled": 100, "time_running": 100},
            {"tid": 1, "event": "cache_references", "raw": 20, "time_enabled": 100, "time_running": 100},
            {"tid": 1, "event": "cache_misses", "raw": 5, "time_enabled": 100, "time_running": 100},
        ]
        value = probe.summarize_perf_counts(rows)
        self.assertEqual(value["scaled_totals"]["cycles"], 150)
        self.assertEqual(value["scaled_totals"]["instructions"], 150)
        self.assertEqual(value["ipc"], 1.0)
        self.assertEqual(value["generic_cache_miss_ratio"], 0.25)

    def test_perf_counter_summary_rejects_zero_running_time(self):
        with self.assertRaisesRegex(ValueError, "time_running"):
            probe.summarize_perf_counts(
                [{"tid": 1, "event": "cycles", "raw": 1, "time_enabled": 1, "time_running": 0}]
            )

    def test_perf_targets_are_exactly_the_128_numa_compute_threads(self):
        snapshot = {
            str(index): {
                "name": (
                    f"numa_{2 + index % 4}_m_0"
                    if index <= 4
                    else f"numa_{2 + index % 4}_t_{index}"
                )
            }
            for index in range(1, 129)
        }
        snapshot["500"] = {"name": "python"}
        snapshot["501"] = {"name": "acl_thread"}
        self.assertEqual(len(probe.select_compute_thread_ids(snapshot)), 128)
        snapshot.pop("128")
        with self.assertRaisesRegex(RuntimeError, "128"):
            probe.select_compute_thread_ids(snapshot)


if __name__ == "__main__":
    unittest.main()
