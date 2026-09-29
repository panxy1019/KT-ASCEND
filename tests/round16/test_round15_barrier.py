from __future__ import annotations

import unittest

from round16.common.round15_barrier import evaluate_round15_receipt


GOOD_SHA = "a" * 64


def _receipt(**changes: object) -> dict[str, object]:
    receipt: dict[str, object] = {
        "round15_terminal_status": "ROUND15_COMPLETE_NO_NEW_GAIN",
        "baseline_manifest_sha256": GOOD_SHA,
        "code_patch_dependency_identity": {"source": GOOD_SHA},
        "cleanup_status": "PASS",
        "owned_runtime_processes": [],
        "resource_lease_released": True,
        "baseline_correctness_open": False,
    }
    receipt.update(changes)
    return receipt


class Round15BarrierTests(unittest.TestCase):
    def test_complete_clean_receipt_opens_barrier(self) -> None:
        decision = evaluate_round15_receipt(_receipt())

        self.assertTrue(decision.ready)
        self.assertEqual(decision.reason_codes, ())
        self.assertEqual(decision.selected_baseline_sha256, GOOD_SHA)

    def test_paused_or_running_is_not_terminal(self) -> None:
        for status in (
            "PAUSED",
            "RUNNING",
            "P0_REPAIR_IN_PROGRESS",
            "ROUND15_NOT_COMPLETE",
            "ROUND15_INCOMPLETE",
            "ROUND15_TERMINALIZING",
            "ROUND15_COMPLETE_NO_NEW_GAIN_PENDING_CLEANUP",
        ):
            with self.subTest(status=status):
                decision = evaluate_round15_receipt(
                    _receipt(round15_terminal_status=status)
                )
                self.assertFalse(decision.ready)
                self.assertIn("ROUND15_NOT_TERMINAL", decision.reason_codes)

    def test_frozen_terminal_spellings_are_accepted(self) -> None:
        for status in (
            "ROUND15_COMPLETE_GAIN",
            "ROUND15_COMPLETE_NO_NEW_GAIN",
            "ROUND15_MAINLINE_COMPLETE_GAIN",
            "ROUND15_MAINLINE_COMPLETE_NO_NEW_GAIN",
            "ROUND15_COMPLETE_SIMPLIFIED_DIRECTIONAL_NO_PROMOTION",
            "ROUND15_PARTIAL_WITH_GAPS",
            "ROUND15_BLOCKED_EXTERNAL_RESOURCE",
            "ROUND15_BLOCKED_TECHNICAL_CONTRACT",
        ):
            with self.subTest(status=status):
                self.assertTrue(
                    evaluate_round15_receipt(
                        _receipt(round15_terminal_status=status)
                    ).ready
                )

    def test_cleanup_process_or_lease_failure_keeps_barrier_closed(self) -> None:
        decision = evaluate_round15_receipt(
            _receipt(
                cleanup_status="FAIL",
                owned_runtime_processes=["model-1"],
                resource_lease_released=False,
            )
        )

        self.assertFalse(decision.ready)
        self.assertEqual(
            decision.reason_codes,
            (
                "CLEANUP_NOT_PASS",
                "ROUND15_RUNTIME_STILL_OWNED",
                "RESOURCE_LEASE_NOT_RELEASED",
            ),
        )

    def test_missing_identity_or_open_correctness_issue_is_rejected(self) -> None:
        decision = evaluate_round15_receipt(
            _receipt(
                baseline_manifest_sha256="not-a-sha",
                code_patch_dependency_identity={},
                baseline_correctness_open=True,
            )
        )

        self.assertFalse(decision.ready)
        self.assertIn("BASELINE_MANIFEST_IDENTITY_INVALID", decision.reason_codes)
        self.assertIn("CODE_PATCH_DEPENDENCY_IDENTITY_MISSING", decision.reason_codes)
        self.assertIn("BASELINE_CORRECTNESS_OPEN", decision.reason_codes)

    def test_missing_fields_fail_closed_without_exception(self) -> None:
        decision = evaluate_round15_receipt({})

        self.assertFalse(decision.ready)
        self.assertIn("ROUND15_NOT_TERMINAL", decision.reason_codes)
        self.assertIn("CLEANUP_NOT_PASS", decision.reason_codes)
        self.assertIsNone(decision.selected_baseline_sha256)


if __name__ == "__main__":
    unittest.main()
