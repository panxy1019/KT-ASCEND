import copy
import unittest

from artifacts.round11.compose_round11_t1_hybrid256_repair import FAILED_ERROR, compose


def row(case, phase, *, cancel_status=None):
    item = {
        "case_id": case,
        "phase": phase,
        "state": "PASS",
        "validated": {"finite_output_logprobs": True, "output_ids": [1, 2]},
        "path": {"path": "HYBRID", "request_total_tokens": 256},
    }
    if cancel_status is not None:
        item.pop("state")
        item.pop("validated")
        item.pop("path")
        item["cancel_status"] = cancel_status
    return item


def supervisor(status, requests, driver_exit):
    return {
        "status": status,
        "driver_exit": driver_exit,
        "requests_recorded": requests,
        "generation_driver_starts": 1,
        "owned_container_removed": True,
        "owned_exit_state": {"OOMKilled": False, "Running": False},
        "guard_exit_ack": {"action": "OWNED_EXITED"},
    }


class ComposeRepairTests(unittest.TestCase):
    def setUp(self):
        rows = [row(f"canary-{i}", "canary") for i in range(9)]
        rows += [
            row("edge256", "extended_prefill"),
            row("edge256", "repeatability"),
            row("decode32", "decode_exact"),
            row("decode64", "decode_exact"),
            row("cancel256", "cancel", cancel_status="PENDING_REUSE_VALIDATION"),
            row("canary_math391", "reuse"),
        ]
        self.failed = {
            "schema": 1, "process": "HYBRID256", "status": "FAIL",
            "error": FAILED_ERROR, "generation_attempts": 15, "abort_controls": 1,
            "automatic_retries": 0, "results": rows, "baseline": "HFINAL_V107",
            "experimental_profile": True, "fixture_sha256": "a" * 64,
            "threshold_tokens": 1025, "chunk_tokens": 32,
            "selection_basis": "REQUEST_TOTAL", "profiler": "OFF",
            "schedule": list(range(15)), "started_wall": 1,
        }
        self.repair = {
            "schema": 1, "process": "HYBRID_CANCEL_REPAIR",
            "status": "PASS_ROUND11_T1_PROCESS", "generation_attempts": 2,
            "abort_controls": 1, "automatic_retries": 0,
            "results": [
                row("cancel256", "cancel", cancel_status="PASS_HYBRID_MIDPREFILL_CANCEL_AND_REUSE"),
                row("canary_math391", "reuse"),
            ],
            "cancel_receipts": ["receipt.json"], "ended_wall": 2,
        }

    def test_composite_keeps_physical_budget_and_source_failure_explicit(self):
        result = compose(
            self.failed, supervisor("FAIL", 15, 1),
            self.repair, supervisor("PASS", 2, 0),
        )
        self.assertTrue(result["composite"])
        self.assertEqual(result["physical_generation_attempts"], 17)
        self.assertEqual(result["physical_model_processes"], 2)
        self.assertEqual(result["composite_validation"]["immutable_failed_source_status"], "FAIL")
        self.assertEqual(len(result["results"]), 15)

    def test_source_error_or_cleanup_drift_fails_closed(self):
        failed = copy.deepcopy(self.failed)
        failed["error"] = "different"
        with self.assertRaisesRegex(ValueError, "failure identity"):
            compose(failed, supervisor("FAIL", 15, 1), self.repair, supervisor("PASS", 2, 0))
        broken = supervisor("PASS", 2, 0)
        broken["owned_container_removed"] = False
        with self.assertRaisesRegex(ValueError, "cleanup"):
            compose(self.failed, supervisor("FAIL", 15, 1), self.repair, broken)


if __name__ == "__main__":
    unittest.main()
