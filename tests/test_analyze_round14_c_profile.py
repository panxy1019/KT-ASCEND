import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "artifacts/round14/analyze_round14_c_profile.py"


def load_module():
    spec = importlib.util.spec_from_file_location("analyze_round14_c_profile", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DEVICE = {"GDN", "QSA_ATTENTION", "QSA_INDEXER", "PLE_CONSUMER", "ROUTED_EXPERT", "NPU_RESIDENT"}


def span(component, layer, device_ms, start, end):
    return {
        "component": component, "layer": layer,
        "start_ns": start, "end_ns": end, "duration_ns": end - start,
        "outcome": "return", "observer_errors": [],
        "device_status": "COMPLETE" if component in DEVICE else "NOT_APPLICABLE",
        "device_ms": device_ms if component in DEVICE else None,
    }


def spans():
    rows = []
    for layer in (0, 24, 46):
        rows.append(span("GDN", layer, 10.0, 0, 100))
    for layer in (3, 23, 47):
        rows.append(span("QSA_ATTENTION", layer, 5.0, 200, 400))
        rows.append(span("QSA_INDEXER", layer, 1.0, 240, 300))
    rows.extend((
        span("PLE_PREPARE", 1, None, 500, 600),
        span("PLE_EXPOSED_WAIT", 1, None, 600, 700),
        span("PLE_CONSUMER", 1, 1.0, 500, 800),
    ))
    for layer in (0, 24, 47):
        rows.append(span("ROUTED_EXPERT", layer, 2.0, 1000, 1400))
        rows.append(span("CPU_SUBMIT", layer, None, 1020, 1080))
        rows.append(span("CPU_RESULT_WAIT", layer, None, 1100, 1300))
        rows.append(span("NPU_RESIDENT", layer, 0.5, 1050, 1200))
    return rows


def final(version, block):
    results = []
    batches = []
    for ordinal, (workload, phase, decode, ttft, e2e) in enumerate((
        ("WARMUP", "warmup", 16, 1.0, 3.0),
        ("W2", "measurement", 128, 1.0, 3.0),
        ("W3", "measurement", 16, 1.5, 3.5),
        ("DRAIN", "drain", 4, 0.5, 1.0),
    ), 1):
        rid = f"v{version}-{workload}"
        results.append({
            "ordinal": ordinal, "workload": workload, "phase": phase,
            "request_id": rid, "completion_tokens": decode,
            "finite_logprobs": True, "finish_reason": "length",
            "output_ids": [3897, 248046] + [1] * max(0, decode - 2),
            "ttft_seconds": ttft, "e2e_seconds": e2e,
            "tpot_seconds": (e2e - ttft) / max(1, decode - 1),
            "qsa": {
                "prefill_strategy": "VECTORIZED",
                "decode_strategy": "DIRECT_BF16_ROW",
                "actual_completion_tokens": decode,
                "first_completion_token_from_prefill_logits": True,
            },
        })
        if phase == "measurement":
            for sequence, mode in enumerate(("extend", "decode"), 1):
                batches.append({
                    "status": "ROUND13_P0_BATCH", "request_id": rid,
                    "mode": mode, "batch_sequence": sequence,
                    "dropped_spans": 0, "observer_errors": [], "spans": spans(),
                })
    return {
        "status": "PASS_ROUND14_C_PROFILE_PROCESS", "version": version,
        "block": block, "parent": "ROUND13_FINAL_S0_UNIFORM_Q32_PROMOTED_QSA",
        "profile_threshold": 513, "default_threshold": 129,
        "profiler": "BOUNDED_OBSERVER_ON", "generation_budget": 4,
        "generation_attempts": 4, "abort_controls": 0, "automatic_retries": 0,
        "fixture_sha256": "f" * 64, "results": results,
        "measurement_requests": [row for row in results if row["phase"] == "measurement"],
        "batches": batches,
        "cleanup": {"guard_exit_ack": True, "container_removed": True, "physical12_after": "EMPTY"},
    }


class Round14CProfileAnalyzerTest(unittest.TestCase):
    def test_two_blocks_scale_samples_and_rank_without_double_counting_children(self):
        api = load_module()
        report = api.analyze([final(274, "A"), final(275, "B")])
        self.assertEqual(report["status"], "PASS_ROUND14_C_FRESH_ATTRIBUTION")
        self.assertEqual(report["process_order"], [274, 275])
        self.assertTrue(report["screen"]["GDN"]["both_blocks"])
        self.assertIn("GDN", report["eligible_hotspots"])
        row = next(item for item in report["requests"] if item["block"] == "A" and item["workload"] == "W2")
        self.assertEqual(row["attributable_ms"]["decode"]["GDN"], 360.0)
        self.assertEqual(row["attributable_ms"]["decode"]["QSA_ATTENTION"], 60.0)
        self.assertIn("parents", report["accounting"])

    def test_rejects_child_outside_parent(self):
        api = load_module()
        a = final(274, "A")
        a["batches"][0]["spans"][4]["end_ns"] = 999
        a["batches"][0]["spans"][4]["duration_ns"] = 759
        with self.assertRaisesRegex(RuntimeError, "nested"):
            api.analyze([a, final(275, "B")])


if __name__ == "__main__":
    unittest.main()
