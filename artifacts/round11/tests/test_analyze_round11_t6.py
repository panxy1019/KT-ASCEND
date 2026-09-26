import copy
import unittest

from artifacts.round11.analyze_round11_t6 import analyze_t6


EXPECTED = {
    147: ("BASELINE", "A", 1),
    148: ("FINAL", "A", 0),
    149: ("FINAL", "B", 1),
    150: ("BASELINE", "B", 0),
}


def performance(index, decode_tokens, elapsed):
    return {
        "index": index,
        "case_id": "edge128",
        "phase": "performance",
        "request_id": f"request-{index}",
        "input_tokens": 128,
        "decode_tokens": decode_tokens,
        "e2e_seconds": elapsed,
        "quality_pass": True,
        "output_ids": list(range(decode_tokens)),
        "token_arrival_monotonic_ns": [1_000_000_000 + i * 400_000_000
                                        for i in range(decode_tokens)],
        "finite_logprobs": True,
        "finish_reason": "length",
        "completion_tokens": decode_tokens,
        "saw_done_marker": True,
        "path": {
            "path": "HYBRID",
            "chunks": [[0, 32], [32, 32], [64, 32], [96, 32]],
            "hybrid_provider_event_count": 32,
        },
    }


def document(version):
    mode, block, aborts = EXPECTED[version]
    warmup = {
        "index": 1,
        "case_id": "canary_math391",
        "phase": "warmup",
        "request_id": "warmup",
        "e2e_seconds": 10.0,
        "quality_pass": True,
        "output_ids": [18, 24, 16, 248046],
        "path": {"path": "HYBRID", "chunks": [[0, 27]],
                 "hybrid_provider_event_count": 8},
    }
    rows = [warmup, performance(2, 32, 20.0 + version % 2),
            performance(3, 64, 30.0 + version % 2),
            performance(4, 128, 50.0 + version % 2)]
    if aborts:
        rows.extend([
            {"index": 5, "case_id": "cancel128", "phase": "cancel",
             "request_id": "cancel", "e2e_seconds": 4.0,
             "cancel_status": "PASS_HYBRID_MIDPREFILL_CANCEL_AND_REUSE"},
            {"index": 6, "case_id": "canary_math391", "phase": "reuse",
             "request_id": "reuse", "e2e_seconds": 2.0,
             "quality_pass": True, "output_ids": [18, 24, 16, 248046],
             "path": {"path": "HYBRID", "chunks": [[0, 27]],
                      "hybrid_provider_event_count": 8}},
        ])
        receipts = ["round11-t6-cancel-receipt.json"]
    else:
        extras = (("canary_marker", "canary_json") if version == 148
                  else ("canary_quote", "canary_transform"))
        for index, case_id in enumerate(extras, 5):
            rows.append({"index": index, "case_id": case_id, "phase": "canary",
                         "request_id": case_id, "e2e_seconds": 3.0,
                         "quality_pass": True, "output_ids": [index, 248046],
                         "path": {"path": "HYBRID", "chunks": [[0, 20]],
                                  "hybrid_provider_event_count": 8}})
        receipts = []
    return {
        "schema": 1,
        "status": "PASS_ROUND11_T6_PROCESS",
        "mode": mode,
        "block": block,
        "version": version,
        "runtime_identity": "ROUND11_HFINAL_EQUALS_V107_ORIGINAL_PROVIDERS",
        "fixture_sha256": "fixture",
        "profiler": "OFF",
        "generation_budget": 6,
        "generation_attempts": 6,
        "abort_controls": aborts,
        "automatic_retries": 0,
        "results": rows,
        "cancel_receipts": receipts,
        "cleanup": {"guard_exit_ack": True, "container_removed": True,
                    "physical12_after": "EMPTY"},
    }


def monitor(version):
    return [
        {"hbm_used_mib": 3124, "owned_process_mib": 0,
         "host_mem_available_kib": 1_900_000_000,
         "numa_meminfo_kib": {str(n): {"MemFree": 120_000_000 + n}
                              for n in (2, 3, 4, 7)},
         "container_status": "created"},
        {"hbm_used_mib": 28000 + version, "owned_process_mib": 26000,
         "host_mem_available_kib": 1_800_000_000,
         "numa_meminfo_kib": {str(n): {"MemFree": 100_000_000 + n}
                              for n in (2, 3, 4, 7)},
         "owned_cgroup": {"memory.usage_in_bytes": "350000000000",
                          "memory.max_usage_in_bytes": "360000000000"}},
        {"action": "OWNED_EXITED_DURING_SAMPLE"},
    ]


class AnalyzeRound11T6Tests(unittest.TestCase):
    def test_valid_abba_same_identity_is_reproducibility_not_speedup(self):
        docs = [document(version) for version in EXPECTED]
        result = analyze_t6(docs, {version: monitor(version) for version in EXPECTED})
        self.assertEqual(result["status"], "PASS_ROUND11_T6_FINAL_INTEGRATION")
        self.assertEqual(result["disposition"], "HFINAL_REQUALIFIED_NO_NEW_PROMOTION")
        self.assertEqual(result["quality"]["decode_lengths"], [32, 64, 128])
        self.assertEqual(result["quality"]["canary_classes"],
                         ["arithmetic", "json", "marker", "quote", "transform"])
        self.assertEqual(result["quality"]["abort_controls"], 2)
        self.assertTrue(result["performance"]["same_compute_identity"])
        self.assertIn("process variance", result["performance"]["interpretation"])
        self.assertEqual(result["resources"]["blocks"]["150"]["max_hbm_used_mib"],
                         28150)

    def test_output_or_path_drift_is_rejected(self):
        docs = [document(version) for version in EXPECTED]
        docs[2]["results"][1]["output_ids"][0] = 999
        with self.assertRaisesRegex(ValueError, "output identity"):
            analyze_t6(docs, {version: monitor(version) for version in EXPECTED})
        docs = [document(version) for version in EXPECTED]
        docs[1]["results"][2]["path"]["path"] = "LAYERWISE"
        with self.assertRaisesRegex(ValueError, "Hybrid path"):
            analyze_t6(docs, {version: monitor(version) for version in EXPECTED})

    def test_cleanup_budget_or_block_identity_failure_is_rejected(self):
        docs = [document(version) for version in EXPECTED]
        docs[0]["cleanup"]["physical12_after"] = "NONEMPTY"
        with self.assertRaisesRegex(ValueError, "cleanup"):
            analyze_t6(docs, {version: monitor(version) for version in EXPECTED})
        docs = [document(version) for version in EXPECTED]
        docs[3]["generation_attempts"] = 7
        with self.assertRaisesRegex(ValueError, "generation contract"):
            analyze_t6(docs, {version: monitor(version) for version in EXPECTED})
        docs = [document(version) for version in EXPECTED]
        docs[2]["block"] = "A"
        with self.assertRaisesRegex(ValueError, "block identity"):
            analyze_t6(docs, {version: monitor(version) for version in EXPECTED})


if __name__ == "__main__":
    unittest.main()
