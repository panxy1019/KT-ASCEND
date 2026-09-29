import unittest

from round17.finalize import aggregate_baseline, baseline_block, qsa_ceiling, summarize_monitor


class FinalizeTests(unittest.TestCase):
    def test_monitor_reports_whole_owned_and_host_peak(self):
        rows = [
            '{"hbm_used_mib":3283}',
            '{"hbm_used_mib":28874,"owned_cgroup":{"memory.max_usage_in_bytes":"367055990784"}}',
        ]
        got = summarize_monitor(rows)
        self.assertEqual(got["whole_card_hbm_peak_mib"], 28874)
        self.assertEqual(got["estimated_owned_hbm_peak_mib"], 25591)
        self.assertGreater(got["host_cgroup_peak_gib"], 341)

    def test_baseline_rejects_truncated_output(self):
        result = {
            "status": "PASS_ROUND16_S0_REQUEST_PERFORMANCE_BLOCK",
            "profiler": "OFF",
            "generation_attempts": 10,
            "results": [
                {"kind": "performance", "completion_tokens": 1, "decode_tokens": 2}
            ] * 9,
        }
        with self.assertRaisesRegex(ValueError, "incomplete"):
            baseline_block(result, ['{"hbm_used_mib":1}'])

    def test_qsa_ceiling_uses_matching_clock_domains(self):
        profile = {
            "host_ms_per_token": {
                "QSA_ATTENTION": {"median": 70.0},
                "QSA_INDEXER": {"median": 35.0},
                "QSA_SPARSE_CORE": {"median": 9.0},
            },
            "device_ms_per_token": {
                "QSA_ATTENTION": {"median": 72.0},
                "QSA_SPARSE_CORE": {"median": 10.0},
            },
            "qsa_uninstrumented_host_ms_per_token": {"median": 19.0},
        }
        got = qsa_ceiling(profile)
        self.assertAlmostEqual(got["perfect_sparse_core_removal_host_percent"], 12.8571, places=3)
        self.assertAlmostEqual(got["perfect_sparse_core_removal_device_percent"], 13.8889, places=3)


if __name__ == "__main__":
    unittest.main()
