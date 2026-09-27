import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "artifacts/round14/compare_round14_a.py"


def load_module():
    spec = importlib.util.spec_from_file_location("compare_round14_a", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def final(version, backend, block, scale, *, native_cleanup=False):
    results = []
    windows = {}
    for workload, tokens in {"W0": 2, "W1": 16, "W2": 128, "W3": 16, "W4": 32}.items():
        results.append({"workload": workload, "phase": "warmup", "e2e_seconds": 99})
        for repeat in range(1, 4):
            results.append({
                "workload": workload, "phase": "measurement", "repeat": repeat,
                "completion_tokens": tokens, "e2e_seconds": scale * repeat,
                "ttft_seconds": scale * repeat / 10,
                "tpot_seconds": scale * repeat / 100 if tokens > 1 else None,
                "finite_logprobs": True, "finish_reason": "length",
            })
        windows[workload] = {
            "seconds": scale * 10,
            "successful_requests": 3,
            "actual_output_tokens": tokens * 3,
            "serial_output_tokens_per_second": tokens * 3 / (scale * 10),
            "requests_per_second": 3 / (scale * 10),
        }
    for index in range(4):
        results.append({"phase": "quality", "case_id": f"q{index}", "quality_pass": True})
    payload = {
        "status": "PASS_ROUND14_A_MODEL_PROCESS", "version": version,
        "backend": backend, "block": block, "generation_attempts": 24,
        "abort_controls": 0, "automatic_retries": 0, "profiler": "OFF",
        "results": results, "measurement_windows": windows,
        "cleanup": {"guard_exit_ack": True, "container_removed": True},
    }
    if native_cleanup:
        payload["cleanup"]["guard_exit_ack"] = {
            "action": "OWNED_EXITED",
            "state": {"Running": False, "Pid": 0},
        }
        payload["resource"] = {
            "physical0_7_processes_after_stop": {str(i): {} for i in range(8)}
        }
    else:
        payload["cleanup"]["physical12_after"] = "EMPTY"
    return payload


class Round14AComparatorTest(unittest.TestCase):
    def test_summary_uses_medians_full_window_and_simultaneous_hbm(self):
        api = load_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            final_path = root / "final.json"
            final_path.write_text(json.dumps(final(235, "KT", "A", 2)))
            monitor = root / "monitor.jsonl"
            monitor.write_text("\n".join(json.dumps(row) for row in [
                {"hbm_used_mib": 20000, "owned_process_mib": 17000,
                 "host_mem_available_kib": 1000,
                 "owned_cgroup": {"memory.current": "100", "memory.max_usage_in_bytes": "150"}},
                {"hbm_used_mib": 21000, "owned_process_mib": 18000,
                 "host_mem_available_kib": 900,
                 "owned_cgroup": {"memory.current": "120", "memory.max_usage_in_bytes": "180"}},
            ]) + "\n")
            summary = api.summarize_process(final_path, monitor)
            self.assertEqual(summary["workloads"]["W2"]["e2e_median_seconds"], 4)
            self.assertEqual(summary["workloads"]["W2"]["actual_output_tokens"], 384)
            self.assertEqual(summary["resources"]["simultaneous_whole_card_hbm_peak_mib"], 21000)
            self.assertEqual(summary["resources"]["host_cgroup_peak_bytes"], 180)

    def test_compare_requires_k_n_n_k_and_reports_system_ratios(self):
        api = load_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = []
            for version, backend, block, scale in (
                (235, "KT", "A", 2),
                (251, "NATIVE_TP8", "A", 1),
                (252, "NATIVE_TP8", "B", 1.2),
                (238, "KT", "B", 2.4),
            ):
                d = root / str(version); d.mkdir()
                p = d / "final.json"; p.write_text(json.dumps(final(
                    version, backend, block, scale,
                    native_cleanup=backend == "NATIVE_TP8",
                )))
                monitor = d / "monitor.jsonl"
                if backend == "KT":
                    rows = [{"hbm_used_mib": 25000, "owned_process_mib": 22000,
                             "host_mem_available_kib": 1000, "owned_cgroup": {"memory.current": "200"}}]
                else:
                    rows = [{"devices": {str(i): {"used_mib": 36000} for i in range(8)},
                             "host_mem_available_kib": 800, "owned_cgroup": {"memory.current": "300"}}]
                monitor.write_text(json.dumps(rows[0]) + "\n")
                inputs.append((p, monitor))
            report = api.compare(inputs)
            self.assertEqual(report["status"], "PASS_ROUND14_A_SYSTEM_COMPARISON")
            self.assertEqual(report["process_order"], [235, 251, 252, 238])
            self.assertAlmostEqual(report["blocks"]["A"]["W2"]["native_over_kt_e2e"], 0.5)
            self.assertAlmostEqual(report["blocks"]["A"]["W2"]["native_over_kt_serial_throughput"], 2.0)
            self.assertEqual(report["processes"]["251"]["resources"]["simultaneous_whole_card_hbm_peak_mib"], 288000)
            self.assertEqual(report["claim_scope"], "SYSTEM_DEPLOYMENT_COMPARISON_NOT_PURE_KT_CAUSAL")

    def test_native_cleanup_requires_structured_ack_and_all_eight_devices_empty(self):
        api = load_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = final(251, "NATIVE_TP8", "A", 1, native_cleanup=True)
            final_path = root / "final.json"
            final_path.write_text(json.dumps(payload))
            monitor = root / "monitor.jsonl"
            monitor.write_text(json.dumps({
                "devices": {str(i): {"used_mib": 36000} for i in range(8)},
                "host_mem_available_kib": 800,
                "owned_cgroup": {"memory.current": "300"},
            }) + "\n")
            summary = api.summarize_process(final_path, monitor)
            self.assertEqual(summary["backend"], "NATIVE_TP8")

            del payload["resource"]["physical0_7_processes_after_stop"]["7"]
            final_path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(RuntimeError, "cleanup contract"):
                api.summarize_process(final_path, monitor)

    def test_native_cleanup_accepts_exit_observed_during_resource_sample(self):
        api = load_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = final(252, "NATIVE_TP8", "B", 1, native_cleanup=True)
            payload["cleanup"]["guard_exit_ack"] = {
                "action": "OWNED_EXITED_DURING_SAMPLE"
            }
            final_path = root / "final.json"
            final_path.write_text(json.dumps(payload))
            monitor = root / "monitor.jsonl"
            monitor.write_text(json.dumps({
                "devices": {str(i): {"used_mib": 36000} for i in range(8)},
                "host_mem_available_kib": 800,
                "owned_cgroup": {"memory.current": "300"},
            }) + "\n")
            summary = api.summarize_process(final_path, monitor)
            self.assertEqual(summary["version"], 252)

    def test_kt_cleanup_requires_physical12_empty(self):
        api = load_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = final(235, "KT", "A", 1)
            payload["cleanup"]["physical12_after"] = "OCCUPIED"
            final_path = root / "final.json"
            final_path.write_text(json.dumps(payload))
            monitor = root / "monitor.jsonl"
            monitor.write_text(json.dumps({
                "hbm_used_mib": 25000, "owned_process_mib": 22000,
                "host_mem_available_kib": 1000,
                "owned_cgroup": {"memory.current": "200"},
            }) + "\n")
            with self.assertRaisesRegex(RuntimeError, "cleanup contract"):
                api.summarize_process(final_path, monitor)


if __name__ == "__main__":
    unittest.main()
