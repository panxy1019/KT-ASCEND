from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from round16.b.q32_hybrid_contract import (
    HybridRouteCase,
    frozen_route_cases,
    validate_layer_bank_catalog,
    validate_v2_receipt,
)


class Q32HybridContractTests(unittest.TestCase):
    def test_route_cases_cover_both_partitions_without_retopk(self):
        cases = frozen_route_cases()
        self.assertEqual(tuple(case.name for case in cases), ("all_npu", "all_cpu", "mixed"))
        self.assertTrue(all(value < 32 for value in cases[0].logical_ids))
        self.assertTrue(all(value >= 32 for value in cases[1].logical_ids))
        self.assertEqual(cases[2].logical_ids, (0, 32, 1, 33, 2, 34, 3, 35, 4, 36))
        with self.assertRaises(ValueError):
            HybridRouteCase("mixed", tuple(range(10)))

    def test_v2_receipt_rejects_unverified_or_wrong_map(self):
        payload = {
            "schema": 1,
            "status": "REAL_Q32_LOADER_AND_LAYOUT_VERIFIED",
            "stage": "V2",
            "device_executed": True,
            "logical_experts": 512,
            "physical_npu_slots": 32,
            "construction_method": "UnquantizedFusedMoEMethod",
            "execution_method": "AscendUnquantizedFusedMoEMethod",
            "postload_order": "UPSTREAM_CREATE_LOAD_THEN_ASCEND_POSTLOAD",
            "load_mode": "SAFETENSORS_CPU_ROW_SLICE_NATIVE_PARAMETER_LOADER",
            "w13_npu_format": 29,
            "w2_npu_format": 29,
            "hidden_full512_npu_payload": False,
            "hybrid_execution": "NOT_RUN_V3",
            "resident_logical_ids": list(range(32)),
            "projection_coverage_count": 96,
            "exact_value_checks": {"gate": True, "up": True, "down": True},
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "v2.json"
            path.write_text(json.dumps(payload))
            observed, digest = validate_v2_receipt(path)
            self.assertEqual(observed["physical_npu_slots"], 32)
            self.assertEqual(len(digest), 64)
            payload["resident_logical_ids"][-1] = 511
            path.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                validate_v2_receipt(path)

    def test_catalog_validation_uses_frozen_full_sha_and_live_size(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bank = root / "layer-00-mxfp4.gguf"
            bank.write_bytes(b"bounded")
            catalog = root / "catalog.json"
            catalog.write_text(
                json.dumps(
                    {
                        "storage": "MXFP4",
                        "banks": [
                            {
                                "layer": 0,
                                "file": bank.name,
                                "storage": "MXFP4",
                                "complete_cpu_backing": True,
                                "logical_ids": list(range(512)),
                                "geometry": {
                                    "experts": 512,
                                    "hidden": 2560,
                                    "intermediate": 640,
                                },
                                "bytes": bank.stat().st_size,
                                "sha256": "a" * 64,
                                "source_identity": {
                                    "revision": "de4b8e4d43b917e7706784d8bb445c9af86a3540"
                                },
                            }
                        ],
                    }
                )
            )
            receipt = validate_layer_bank_catalog(catalog, bank, layer_id=0)
            self.assertEqual(receipt["bank_identity_method"], "TRUSTED_FULL_SHA_RECEIPT_PLUS_LIVE_SIZE")
            bank.write_bytes(b"changed size")
            with self.assertRaises(ValueError):
                validate_layer_bank_catalog(catalog, bank, layer_id=0)


if __name__ == "__main__":
    unittest.main()
