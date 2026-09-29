from __future__ import annotations

import copy
import unittest


def _module():
    from round16.common import perf_contract

    return perf_contract


def _workload(name, requested, *, ttft, tpot, e2e, window_seconds):
    return {
        "workload": name,
        "warmup_attempts": 1,
        "measurement_attempts": 2,
        "requested_output_tokens": requested,
        "actual_output_tokens": [requested, requested],
        "ttft_ms": [ttft, ttft],
        "tpot_ms": [tpot, tpot],
        "e2e_ms": [e2e, e2e],
        "measurement_window_seconds": window_seconds,
        "measurement_window_output_tokens": requested * 2,
        "measurement_window_successes": 2,
        "measurement_window_failures": 0,
    }


def _block(role, process_id, *, candidate=False):
    factor = 0.88 if candidate else 1.0
    throughput_factor = 1.12 if candidate else 1.0
    return {
        "role": role,
        "process_id": process_id,
        "common_identity_sha256": "a" * 64,
        "model_identity_sha256": ("c" if candidate else "b") * 64,
        "profiler_enabled": False,
        "quality_pass": True,
        "state_pass": True,
        "cancel_reuse_pass": True,
        "resource_pass": True,
        "canary_pass": True,
        "error_count": 0,
        "generation_attempts": 10,
        "workloads": {
            "W1": _workload(
                "W1", 16, ttft=100 * factor, tpot=20, e2e=400 * factor,
                window_seconds=(32 / (2.0 * throughput_factor)),
            ),
            "W2": _workload(
                "W2", 128, ttft=100, tpot=100 * factor, e2e=12800 * factor,
                window_seconds=(256 / (3.0 * throughput_factor)),
            ),
            "W4": _workload(
                "W4", 32, ttft=80, tpot=80, e2e=2560,
                window_seconds=(64 / (2.5 * throughput_factor)),
            ),
        },
    }


def _document():
    return {
        "schema": 1,
        "comparison_id": "round16-a-request-abba",
        "comparison_kind": "A_SAME_RUNTIME",
        "common_identity_sha256": "a" * 64,
        "target_contract": {
            "target": "DECODE",
            "frozen_before_first_process": True,
            "primary_workload": "W2",
            "primary_latency_metric": "TPOT",
            "primary_throughput_metric": "CONTINUOUS_WINDOW_OUTPUT_TOKENS_PER_SECOND",
        },
        "blocks": [
            _block("CONTROL", "control-a"),
            _block("CANDIDATE", "candidate-a", candidate=True),
            _block("CANDIDATE", "candidate-b", candidate=True),
            _block("CONTROL", "control-b"),
        ],
    }


class PerformanceContractTests(unittest.TestCase):
    def test_decode_abba_qualifies_only_from_continuous_window_throughput(self):
        result = _module().evaluate_abba(_document())
        self.assertEqual(result.status, "QUALIFIED_REQUEST_GAIN")
        self.assertTrue(result.promoted)
        self.assertEqual(len(result.pairs), 2)
        for pair in result.pairs:
            self.assertLessEqual(pair.primary_latency_ratio, 0.90)
            self.assertGreaterEqual(pair.primary_throughput_ratio, 1.10)

    def test_incomplete_or_reused_process_block_is_not_a_performance_negative(self):
        module = _module()
        for mutate in (
            lambda d: d["blocks"].pop(),
            lambda d: d["blocks"][2].__setitem__("process_id", "candidate-a"),
            lambda d: d["blocks"][0].__setitem__("role", "CANDIDATE"),
            lambda d: d["blocks"][1].__setitem__("generation_attempts", 9),
        ):
            document = _document()
            mutate(document)
            with self.subTest(document=document):
                with self.assertRaises(module.ComparisonValidationError):
                    module.evaluate_abba(document)

    def test_missing_tokens_failures_or_profiler_data_are_rejected_before_speed(self):
        module = _module()
        document = _document()
        document["blocks"][1]["workloads"]["W2"]["actual_output_tokens"][1] = 127
        with self.assertRaises(module.ComparisonValidationError):
            module.evaluate_abba(document)

        document = _document()
        document["blocks"][1]["workloads"]["W2"]["measurement_window_failures"] = 1
        with self.assertRaises(module.ComparisonValidationError):
            module.evaluate_abba(document)

        document = _document()
        document["blocks"][1]["profiler_enabled"] = True
        with self.assertRaises(module.ComparisonValidationError):
            module.evaluate_abba(document)

    def test_throughput_is_not_derived_from_inverse_tpot(self):
        document = _document()
        for index in (1, 2):
            # Excellent TPOT, but the complete request window is slower.
            document["blocks"][index]["workloads"]["W2"]["tpot_ms"] = [50, 50]
            document["blocks"][index]["workloads"]["W2"][
                "measurement_window_seconds"
            ] = 100.0
        result = _module().evaluate_abba(document)
        self.assertEqual(result.status, "MEASURED_NO_PROMOTION")
        self.assertFalse(result.promoted)
        self.assertIn("PRIMARY_THROUGHPUT_GATE_FAILED", result.reasons)

    def test_non_primary_protection_and_correctness_are_distinct_dispositions(self):
        document = _document()
        for index in (1, 2):
            document["blocks"][index]["workloads"]["W4"]["e2e_ms"] = [3000, 3000]
        result = _module().evaluate_abba(document)
        self.assertEqual(result.status, "MEASURED_NO_PROMOTION")
        self.assertIn("PROTECTION_GATE_FAILED", result.reasons)

        document = _document()
        document["blocks"][1]["quality_pass"] = False
        result = _module().evaluate_abba(document)
        self.assertEqual(result.status, "REJECTED_CORRECTNESS_OR_RESOURCE")
        self.assertFalse(result.promoted)

    def test_target_must_be_frozen_before_any_process(self):
        module = _module()
        document = _document()
        document["target_contract"]["frozen_before_first_process"] = False
        with self.assertRaises(module.ComparisonValidationError):
            module.evaluate_abba(document)

    def test_prefill_gate_is_frozen_and_uses_w1_ttft_and_e2e(self):
        document = _document()
        document["target_contract"] = {
            "target": "PREFILL",
            "frozen_before_first_process": True,
            "primary_workload": "W1",
            "primary_latency_metric": "TTFT_AND_E2E",
            "primary_throughput_metric": "NONE",
        }
        for index in (1, 2):
            w1 = document["blocks"][index]["workloads"]["W1"]
            w1["ttft_ms"] = [84, 84]
            w1["e2e_ms"] = [89, 89]
        result = _module().evaluate_abba(document)
        self.assertEqual(result.status, "QUALIFIED_REQUEST_GAIN")


if __name__ == "__main__":
    unittest.main()
