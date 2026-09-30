from __future__ import annotations

import unittest

from round18.native_partial_contract import (
    NativePartialCase,
    frozen_native_partial_cases,
    validate_native_partial_receipt,
)


class NativePartialContractTests(unittest.TestCase):
    def test_matrix_covers_resident_cpu_mixed_multi_and_ties(self) -> None:
        cases = frozen_native_partial_cases()
        names = {case.name for case in cases}
        self.assertEqual(
            names,
            {
                "all_resident",
                "all_nonresident",
                "mixed_1_9",
                "mixed_5_5",
                "mixed_9_1",
                "multi_token",
                "topk_ties",
            },
        )
        self.assertEqual(sum(expert < 32 for expert in cases[2].rows[0]), 1)
        self.assertEqual(sum(expert < 32 for expert in cases[4].rows[0]), 9)
        self.assertEqual(len(cases[5].rows), 3)
        self.assertTrue(cases[6].tie_topk)

    def test_duplicate_or_out_of_domain_route_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "distinct"):
            NativePartialCase("bad", ((0, 0, 1, 2, 3, 4, 5, 6, 7, 8),))
        with self.assertRaisesRegex(ValueError, "logical domain"):
            NativePartialCase("bad", ((0, 1, 2, 3, 4, 5, 6, 7, 8, 512),))

    def test_receipt_requires_native_selector_once_and_all_cases(self) -> None:
        receipt = {
            "status": "PASS_NATIVE_RESIDENT_PARTIAL",
            "device_executed": True,
            "native_runner_preserved": True,
            "native_quant_method": "AscendUnquantizedFusedMoEMethod",
            "logical_experts": 512,
            "physical_npu_slots": 32,
            "selector_calls_per_execution": 1,
            "route_weight_application": "EXACTLY_ONCE",
            "shared_finalize_residual_in_partial": False,
            "cases": [
                {"case": case.name, "passed": True, "repeat_exact": True}
                for case in frozen_native_partial_cases()
            ],
        }
        validate_native_partial_receipt(receipt)
        receipt["selector_calls_per_execution"] = 2
        with self.assertRaisesRegex(ValueError, "selector"):
            validate_native_partial_receipt(receipt)


if __name__ == "__main__":
    unittest.main()
