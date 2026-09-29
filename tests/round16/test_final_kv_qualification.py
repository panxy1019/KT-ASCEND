from __future__ import annotations

import json
from pathlib import Path
import unittest


class FinalKvQualificationTests(unittest.TestCase):
    def test_single_page_shape_patch_is_not_the_qualification_evidence(self) -> None:
        root = Path(__file__).resolve().parents[2]
        qualification = json.loads(
            (
                root
                / "artifacts/round16/ROUND16_B_KV_QUALIFICATION.json"
            ).read_text()
        )
        device = json.loads(
            (
                root
                / "artifacts/round16-b/evidence/"
                "20260929-round16-b-qsa-multihead-v001/result.json"
            ).read_text()
        )

        self.assertFalse(qualification["single_page_unsqueeze_only_is_sufficient"])
        self.assertEqual(device["status"], "PASS_QSA_MULTIPAGE_MULTIHEAD_DEVICE")
        self.assertEqual(device["cache_shape"], [3, 3072, 2, 256])
        self.assertEqual(device["block_table"], [[2, 0, 1]])
        self.assertEqual(device["pages_touched"], [0, 1, 2])
        self.assertEqual(device["query_heads"], 24)
        self.assertEqual(device["kv_heads"], 2)
        self.assertTrue(device["store_exact"])
        self.assertEqual(
            qualification["qualification_boundary"][
                "full_model_long_context_cross_page_generation"
            ],
            "NOT_RUN_OUTSIDE_ROUND16_SCOPE",
        )
        self.assertEqual(
            qualification["qualification_boundary"]["broad_kv_correctness"],
            "NOT_CLAIMED",
        )


if __name__ == "__main__":
    unittest.main()
