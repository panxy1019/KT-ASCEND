import csv
import json
from pathlib import Path
import tempfile
import unittest


from tools.kt_perf_closure.build_token_accounting import build_accounting


class BuildTokenAccountingTest(unittest.TestCase):
    def test_partitions_top_level_host_wall_without_nested_indexer(self):
        analysis = {
            "status": "PASS_KT_ROOT_NATIVE_HOST_JOIN",
            "batch_rows": [
                {
                    "batch_sequence": 6,
                    "scheduler_wall_ns": 100_000_000,
                    "routed_expert_parent_sum_ns": 30_000_000,
                    "cpu_task_sum_ns_nested_with_npu": 2_000_000,
                    "native_sync_wait_sum_ns": 100_000,
                }
            ],
        }
        spans = [
            {
                "status": "ROUND13_P0_BATCH",
                "request_id": "x:W2_MEASUREMENT",
                "mode": "decode",
                "batch_sequence": 6,
                "spans": [
                    *(
                        {"component": "GDN", "duration_ns": 1_000_000, "layer": i}
                        for i in range(36)
                    ),
                    *(
                        {"component": "QSA_ATTENTION", "duration_ns": 2_000_000, "layer": i}
                        for i in range(12)
                    ),
                    *(
                        {"component": "QSA_INDEXER", "duration_ns": 1_500_000, "layer": i}
                        for i in range(12)
                    ),
                    {"component": "PLE_CONSUMER", "duration_ns": 1_000_000, "layer": 1},
                ],
            }
        ]

        result = build_accounting(analysis, spans)

        row = result["token_rows"][0]
        self.assertEqual(row["gdn_ns"], 36_000_000)
        self.assertEqual(row["qsa_attention_ns"], 24_000_000)
        self.assertEqual(row["ple_consumer_ns"], 1_000_000)
        self.assertEqual(row["routed_expert_ns"], 30_000_000)
        self.assertEqual(row["other_dense_dispatch_ns"], 9_000_000)
        self.assertNotIn("qsa_indexer_ns", row)
        self.assertEqual(result["coverage"]["gdn_layers_per_token"], 36)
        self.assertEqual(result["coverage"]["qsa_layers_per_token"], 12)
        self.assertEqual(result["scope"], "EXACT_ALL_LAYER_HOST_PARENT_PARTITION")

    def test_rejects_missing_full_layer_coverage(self):
        analysis = {
            "status": "PASS_KT_ROOT_NATIVE_HOST_JOIN",
            "batch_rows": [
                {
                    "batch_sequence": 6,
                    "scheduler_wall_ns": 10,
                    "routed_expert_parent_sum_ns": 1,
                    "cpu_task_sum_ns_nested_with_npu": 1,
                    "native_sync_wait_sum_ns": 0,
                }
            ],
        }
        spans = [{
            "status": "ROUND13_P0_BATCH",
            "request_id": "x:W2_MEASUREMENT",
            "mode": "decode",
            "batch_sequence": 6,
            "spans": [],
        }]
        with self.assertRaisesRegex(ValueError, "36 GDN"):
            build_accounting(analysis, spans)


if __name__ == "__main__":
    unittest.main()
