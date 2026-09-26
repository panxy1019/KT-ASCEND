import copy
import unittest

from artifacts.round11.analyze_round11_t5 import rank_b_route_hotspots


GDN_LAYERS = (0, 24, 46)
QSA_LAYERS = (3, 23, 47)


def span(component, layer, start, end, device_ms=None):
    return {
        "component": component,
        "layer": layer,
        "start_ns": start,
        "end_ns": end,
        "duration_ns": end - start,
        "device_ms": device_ms,
        "device_status": "COMPLETE" if device_ms is not None else "NOT_APPLICABLE",
        "observer_errors": [],
    }


def batch(request_id, mode, *, gdn_ms=10.0, qsa_ms=30.0, indexer_ms=10.0,
          ple_wait_ms=100.0):
    spans = []
    cursor = 1_000_000
    for layer in GDN_LAYERS:
        spans.append(span("GDN", layer, cursor, cursor + 10_000, gdn_ms))
        cursor += 20_000
    for layer in QSA_LAYERS:
        parent_start = cursor
        parent_end = cursor + 20_000
        spans.append(span("QSA_ATTENTION", layer, parent_start, parent_end, qsa_ms))
        spans.append(span("QSA_INDEXER", layer, parent_start + 2_000,
                          parent_end - 2_000, indexer_ms))
        cursor += 30_000
    spans.extend([
        span("PLE_PREPARE", 1, cursor, cursor + 5_000),
        span("PLE_EXPOSED_WAIT", 1, cursor + 5_000,
             cursor + 5_000 + int(ple_wait_ms * 1e6)),
        span("PLE_CONSUMER", 1, cursor + 10_000,
             cursor + 20_000, 2.0),
    ])
    return {
        "request_id": request_id,
        "mode": mode,
        "batch_sequence": 0 if mode == "extend" else 1,
        "spans": spans,
        "dropped_spans": 0,
        "observer_errors": [],
    }


def process(block, *, gdn_ms=10.0, qsa_ms=30.0, indexer_ms=10.0,
            ple_wait_ms=100.0, e2e_seconds=10.0):
    request_id = f"r11-t5-{block.lower()}-measurement"
    return {
        "schema": 1,
        "status": "PASS_ROUND11_T5_PROCESS",
        "block": block,
        "parent_identity": "HFINAL_OR_PROMOTED_KUNPENG",
        "profiler": "BOUNDED_OBSERVER_ON",
        "bindings": {
            "GDN": list(GDN_LAYERS),
            "QSA_ATTENTION": list(QSA_LAYERS),
            "QSA_INDEXER": list(QSA_LAYERS),
            "PLE": [1],
        },
        "requests": [{
            "request_id": request_id,
            "phase": "measurement",
            "input_tokens": 64,
            "decode_tokens": 32,
            "e2e_seconds": e2e_seconds,
            "quality_pass": True,
        }],
        "batches": [
            batch(request_id, "extend", gdn_ms=gdn_ms, qsa_ms=qsa_ms,
                  indexer_ms=indexer_ms, ple_wait_ms=ple_wait_ms),
            batch(request_id, "decode", gdn_ms=gdn_ms, qsa_ms=qsa_ms,
                  indexer_ms=indexer_ms, ple_wait_ms=ple_wait_ms),
        ],
        "cleanup": {
            "guard_exit_ack": True,
            "container_removed": True,
            "physical12_after": "EMPTY",
        },
    }


class AnalyzeRound11T5Tests(unittest.TestCase):
    def test_nested_qsa_is_not_double_counted_and_no_hotspot_is_terminal(self):
        result = rank_b_route_hotspots([process("A"), process("B")])
        self.assertEqual(result["status"], "PASS_ROUND11_T5_HOTSPOT_RANKING")
        self.assertEqual(result["disposition"], "NO_B_ROUTE_PERFORMANCE_CANDIDATE")
        # Per batch: median parent 30 ms * 12 actual layers. Two batches.
        self.assertAlmostEqual(result["module_medians"]["QSA"]["attributable_ms"], 720.0)
        # Indexer is a breakdown of QSA parent, never added to QSA total.
        self.assertAlmostEqual(result["module_medians"]["QSA"]["indexer_device_ms"], 240.0)
        self.assertAlmostEqual(result["module_medians"]["QSA"]["exclusive_remainder_ms"], 480.0)

    def test_gdn_crossing_ten_percent_selects_the_only_candidate(self):
        result = rank_b_route_hotspots([
            process("A", gdn_ms=50.0), process("B", gdn_ms=50.0)
        ])
        # 50 ms * 36 layers * two batches = 3.6 s of a 10 s request.
        self.assertGreaterEqual(result["module_medians"]["GDN"]["critical_fraction"], 0.10)
        self.assertEqual(result["disposition"], "B_ROUTE_CANDIDATE_GDN")
        self.assertEqual(result["candidate"], "GDN")

    def test_ple_exposed_wait_crossing_five_percent_selects_ple(self):
        result = rank_b_route_hotspots([
            process("A", ple_wait_ms=300.0), process("B", ple_wait_ms=300.0)
        ])
        self.assertAlmostEqual(result["module_medians"]["PLE"]["exposed_wait_fraction"], 0.06)
        self.assertEqual(result["disposition"], "B_ROUTE_CANDIDATE_PLE")

    def test_incomplete_device_or_cleanup_evidence_is_rejected(self):
        runs = [process("A"), process("B")]
        runs[0]["batches"][0]["spans"][0]["device_status"] = "PENDING"
        with self.assertRaisesRegex(ValueError, "device timing"):
            rank_b_route_hotspots(runs)
        runs = [process("A"), process("B")]
        runs[1]["cleanup"]["physical12_after"] = "NONEMPTY"
        with self.assertRaisesRegex(ValueError, "cleanup"):
            rank_b_route_hotspots(runs)

    def test_missing_sample_layer_or_parent_identity_is_rejected(self):
        runs = [process("A"), process("B")]
        runs[0]["batches"][0]["spans"] = [
            row for row in runs[0]["batches"][0]["spans"]
            if not (row["component"] == "GDN" and row["layer"] == 24)
        ]
        with self.assertRaisesRegex(ValueError, "sample coverage"):
            rank_b_route_hotspots(runs)
        runs = [process("A"), process("B")]
        runs[1]["parent_identity"] = "OTHER"
        with self.assertRaisesRegex(ValueError, "parent"):
            rank_b_route_hotspots(runs)


if __name__ == "__main__":
    unittest.main()
