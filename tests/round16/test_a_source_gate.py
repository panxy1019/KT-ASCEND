from __future__ import annotations

import unittest

from round16.a.fresh_a1_profile import RUNTIME_VERSION
from round16.a.source_gate import SourceDeltaError, require_parent_or_bound_delta


RELATIVE = "sglang/srt/layers/moe/kt_ep_wrapper.py"
PARENT = "a" * 64
CURRENT = "b" * 64


def binding() -> dict:
    return {
        "round16_a1_profile": {
            "schema": 1,
            "status": "INSTALLED_DEFAULT_OFF_PROFILE_ONLY",
            "runtime_version": RUNTIME_VERSION,
            "default_enabled": False,
            "compute_semantics": "UNCHANGED_OBSERVER_ONLY",
            "parent_overlay_files": {RELATIVE: PARENT},
            "modified_overlay_files": {RELATIVE: CURRENT},
        }
    }


class SourceGateTests(unittest.TestCase):
    def test_exact_parent_source_remains_accepted(self) -> None:
        self.assertEqual(
            require_parent_or_bound_delta(
                relative=RELATIVE,
                expected_parent_sha256=PARENT,
                actual_sha256=PARENT,
                overlay_receipt={},
            ),
            "BASELINE_EXACT",
        )

    def test_exact_default_off_observer_delta_is_accepted(self) -> None:
        self.assertEqual(
            require_parent_or_bound_delta(
                relative=RELATIVE,
                expected_parent_sha256=PARENT,
                actual_sha256=CURRENT,
                overlay_receipt=binding(),
            ),
            "ROUND16_BOUND_DELTA",
        )

    def test_unknown_or_drifted_delta_is_rejected(self) -> None:
        cases = [
            ("other.py", PARENT, CURRENT, binding()),
            (RELATIVE, "c" * 64, CURRENT, binding()),
            (RELATIVE, PARENT, "d" * 64, binding()),
            (
                RELATIVE,
                PARENT,
                CURRENT,
                {
                    **binding(),
                    "round16_a1_profile": {
                        **binding()["round16_a1_profile"],
                        "default_enabled": True,
                    },
                },
            ),
        ]
        for relative, parent, current, receipt in cases:
            with self.subTest(relative=relative, parent=parent, current=current):
                with self.assertRaises(SourceDeltaError):
                    require_parent_or_bound_delta(
                        relative=relative,
                        expected_parent_sha256=parent,
                        actual_sha256=current,
                        overlay_receipt=receipt,
                    )


if __name__ == "__main__":
    unittest.main()
