import json
from pathlib import Path
import tempfile
import unittest

from artifacts.round11.analyze_round11_t4 import calculate_break_even


def process(mode, block, interval_s, migration_s=None):
    rows = []
    for length in (32, 64, 128, 256):
        arrivals = [1_000_000_000]
        for _ in range(1, length):
            arrivals.append(arrivals[-1] + int(interval_s * 1e9))
        row = {
            "decode_tokens": length,
            "input_tokens": 64,
            "output_ids": list(range(length)),
            "token_arrival_monotonic_ns": arrivals,
            "finite_logprobs": True,
            "finish_reason": "length",
            "path": "HYBRID",
            "e2e_seconds": length * interval_s + (migration_s or 0),
        }
        if mode == "DYNAMIC":
            row["migration"] = {
                "status": "PREFILL_END_COMMIT",
                "live_state_preserved": True,
                "changed_layers": 48,
                "copied_experts": 96,
                "timing_ns": {
                    "route_statistics": 100_000_000,
                    "consumer_drain_sync": 200_000_000,
                    "state_fence_before": 100_000_000,
                    "selection": 100_000_000,
                    "migration_copy_and_sync": int(migration_s * 1e9) - 700_000_000,
                    "mapping_publication": 50_000_000,
                    "state_fence_after": 50_000_000,
                    "cleanup": 100_000_000,
                    "total_exposed": int(migration_s * 1e9),
                },
            }
        rows.append(row)
    return {
        "schema": 1,
        "status": "PASS_ROUND11_T4_PROCESS",
        "mode": mode,
        "block": block,
        "profiler": "OFF",
        "parent_identity": "HFINAL_OR_PROMOTED_KUNPENG",
        "results": rows,
        "cleanup": {
            "guard_exit_ack": True,
            "container_removed": True,
            "physical12_after": "EMPTY",
        },
    }


class AnalyzeRound11T4Tests(unittest.TestCase):
    def test_finite_break_even_uses_steady_tail_and_full_exposed_migration(self):
        runs = [
            process("STATIC", "A", 0.5),
            process("DYNAMIC", "A", 0.4, 10.0),
            process("DYNAMIC", "B", 0.4, 10.0),
            process("STATIC", "B", 0.5),
        ]
        result = calculate_break_even(runs)
        self.assertEqual(result["status"], "PASS_ROUND11_T4_BREAK_EVEN")
        self.assertAlmostEqual(result["S_static_seconds_per_token"], 0.5)
        self.assertAlmostEqual(result["D_dynamic_seconds_per_token"], 0.4)
        self.assertAlmostEqual(result["M_migration_exposed_seconds"], 10.0)
        self.assertAlmostEqual(result["N_break_even_tokens"], 100.0)
        self.assertEqual(result["disposition"], "FINITE_BREAK_EVEN_CONFIRMED_AT_128")
        self.assertEqual(result["request_confirmed_at_tokens"], 128)

    def test_dynamic_not_faster_has_no_finite_break_even(self):
        runs = [
            process("STATIC", "A", 0.4),
            process("DYNAMIC", "A", 0.5, 10.0),
            process("DYNAMIC", "B", 0.5, 10.0),
            process("STATIC", "B", 0.4),
        ]
        result = calculate_break_even(runs)
        self.assertIsNone(result["N_break_even_tokens"])
        self.assertEqual(result["disposition"], "NO_FINITE_BREAK_EVEN")

    def test_rejects_missing_migration_phase_or_cleanup(self):
        runs = [
            process("STATIC", "A", 0.5),
            process("DYNAMIC", "A", 0.4, 10.0),
            process("DYNAMIC", "B", 0.4, 10.0),
            process("STATIC", "B", 0.5),
        ]
        del runs[1]["results"][0]["migration"]["timing_ns"]["selection"]
        with self.assertRaisesRegex(ValueError, "migration phase"):
            calculate_break_even(runs)
        runs[1]["results"][0]["migration"]["timing_ns"]["selection"] = 100_000_000
        runs[3]["cleanup"]["physical12_after"] = "NONEMPTY"
        with self.assertRaisesRegex(ValueError, "cleanup"):
            calculate_break_even(runs)

    def test_requires_exact_repeat_output_ids_within_each_mode(self):
        runs = [
            process("STATIC", "A", 0.5),
            process("DYNAMIC", "A", 0.4, 10.0),
            process("DYNAMIC", "B", 0.4, 10.0),
            process("STATIC", "B", 0.5),
        ]
        runs[2]["results"][2]["output_ids"][-1] = 999
        with self.assertRaisesRegex(ValueError, "same-mode output"):
            calculate_break_even(runs)

        runs[2]["results"][2]["output_ids"][-1] = 127
        result = calculate_break_even(runs)
        self.assertTrue(result["quality"]["same_mode_output_ids_exact"])


if __name__ == "__main__":
    unittest.main()
