import json
import tempfile
import unittest
from pathlib import Path

from artifacts.round11.compose_round11_t1_hybrid1024_cutoff import compose


def passing_768_row():
    return {
        "case_id": "edge768",
        "phase": "extended_prefill",
        "state": "PASS",
        "e2e_seconds": 282.0,
        "validated": {
            "output_ids": [3897, 248046],
            "finite_output_logprobs": True,
            "normal_eos": True,
        },
        "path": {
            "path": "HYBRID",
            "request_total_tokens": 768,
            "actual_dispatch_tokens": [32] * 24,
            "hybrid_provider_event_count": 192,
            "layerwise_event_count": 0,
        },
    }


class ComposeHybrid1024CutoffTests(unittest.TestCase):
    def setUp(self):
        self.result = {
            "schema": 1,
            "status": "FAIL",
            "process": "HYBRID1024",
            "generation_budget": 2,
            "generation_attempts": 2,
            "abort_controls": 0,
            "automatic_retries": 0,
            "results": [passing_768_row()],
            "error": "RemoteDisconnected('Remote end closed connection without response')",
        }
        state = {
            "Status": "exited",
            "Running": False,
            "OOMKilled": False,
            "ExitCode": 137,
        }
        self.supervisor = {
            "status": "FAIL",
            "process": "HYBRID1024",
            "version": 111,
            "container_id": "a" * 64,
            "generation_driver_starts": 1,
            "error": "RuntimeError('server exited before Round11 T1 completion')",
            "owned_exit_state": state,
            "guard_exit_ack": {
                "action": "OWNED_EXITED",
                "container_id": "a" * 64,
                "state": state,
                "after_guard_stop": True,
            },
            "owned_container_removed": True,
        }
        self.monitor = [
            {
                "time": 1,
                "hbm_used_mib": 28772,
                "owned_process_mib": 25756,
                "host_mem_available_kib": 1700 * 1024**2,
                "owned_cgroup": {"memory.events": "low 0\nhigh 0\noom 0"},
            },
            {
                "time": 2,
                "action": "STOP_OWNED",
                "reason": "NUMA4 free and conservative reclaim headroom below reserve",
            },
            {
                "time": 3,
                "action": "OWNED_EXITED",
                "container_id": "a" * 64,
                "state": state,
                "after_guard_stop": True,
            },
        ]

    def _write(self, root, name, value, *, jsonl=False):
        path = Path(root) / name
        if jsonl:
            path.write_text("".join(json.dumps(row) + "\n" for row in value))
        else:
            path.write_text(json.dumps(value) + "\n")
        return path

    def test_compose_preserves_raw_failure_and_sets_qualified_upper_bound(self):
        with tempfile.TemporaryDirectory() as root:
            result = self._write(root, "result.json", self.result)
            supervisor = self._write(root, "supervisor.json", self.supervisor)
            monitor = self._write(root, "monitor.jsonl", self.monitor, jsonl=True)
            cleanup = Path(root) / "post-cleanup-proc-mem.txt"
            cleanup.write_text("No process in device.\n")
            composed = compose(result, supervisor, monitor, cleanup)
        self.assertEqual(composed["status"], "PASS_ROUND11_T1_PROCESS_WITH_RESOURCE_CUTOFF")
        self.assertEqual(composed["raw_source_status"], "FAIL")
        self.assertEqual(composed["hybrid_qualified_up_to_tokens"], 768)
        self.assertEqual(composed["resource_cutoff_tokens"], 1024)
        self.assertEqual(len(composed["source_hashes"]), 4)
        self.assertEqual(composed["results"], [passing_768_row()])

    def test_rejects_ordinary_server_failure_without_guard_reason(self):
        self.monitor[1]["reason"] = "telemetry/guard failure"
        with tempfile.TemporaryDirectory() as root:
            result = self._write(root, "result.json", self.result)
            supervisor = self._write(root, "supervisor.json", self.supervisor)
            monitor = self._write(root, "monitor.jsonl", self.monitor, jsonl=True)
            with self.assertRaisesRegex(ValueError, "resource cutoff"):
                cleanup = Path(root) / "post-cleanup-proc-mem.txt"
                cleanup.write_text("No process in device.\n")
                compose(result, supervisor, monitor, cleanup)


if __name__ == "__main__":
    unittest.main()
