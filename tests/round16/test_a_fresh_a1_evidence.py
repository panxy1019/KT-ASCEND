from __future__ import annotations

import copy
import unittest

from round16.a.fresh_a1_analysis import analysis_payload
from round16.a.fresh_a1_evidence import (
    EvidenceNormalizationError,
    normalize_observer_rows,
)
from tests.round16.test_a_fresh_a1_analysis import (
    EXPERT_LAYERS,
    _block,
    _event,
)


RUNTIME_BINDINGS = {
    "status": "ROUND16_A1_OBSERVER_BOUND",
    "host_layers": list(range(48)),
    "device_sample_layers": [0, 24, 47],
    "device_query": "DEFERRED_QUERY_ONLY_NO_SYNCHRONIZE",
}
ROUND13_BINDINGS = {
    "status": "ROUND13_P0_ATTRIBUTION_BOUND",
    "bindings": {
        "GDN": [0, 24, 46],
        "QSA_ATTENTION": [3, 23, 47],
        "QSA_INDEXER": [3, 23, 47],
        "PLE": [1],
        "EXPERT": [0, 24, 47],
    },
    "device_query": "DEFERRED_QUERY_ONLY_NO_SYNCHRONIZE",
}


def _raw_inputs(block: str):
    base = _block(block, gdn_ms=4.0, expert_ms=3.0)
    block_id = base["receipt"]["block_id"]
    runtime_rows = [
        copy.deepcopy(row)
        for row in base["observer"]["rows"]
        if row["component"] not in {"QSA_INDEXER", "GDN", "DECODE_PHASE_WALL"}
    ]
    serial = 100_000
    for generation in base["receipt"]["generations"]:
        request_id = generation["request_id"]
        for layer in sorted(set(range(48)) - set(EXPERT_LAYERS)):
            correlation = f"{request_id}:expert:{layer}"
            serial += 1
            parent = _event(
                block_id=block_id,
                request_id=request_id,
                component="PARENT_ROUTED_EXPERT",
                layer=layer,
                serial=serial,
                clock_domain="HOST_MONOTONIC_NS",
                value=50_000,
                parent_span_id=f"{request_id}:batch",
                correlation_id=correlation,
            )
            runtime_rows.append(parent)
            for component, domain, value in (
                ("HOST_DISPATCH", "HOST_MONOTONIC_NS", 100),
                ("HOST_SYNC", "HOST_MONOTONIC_NS", 200),
                ("HOST_MERGE", "HOST_MONOTONIC_NS", 50),
                ("CPU_EXPERT", "NATIVE_WORKER_CLOCK_UNAVAILABLE", None),
            ):
                serial += 1
                runtime_rows.append(
                    _event(
                        block_id=block_id,
                        request_id=request_id,
                        component=component,
                        layer=layer,
                        serial=serial,
                        clock_domain=domain,
                        value=value,
                        parent_span_id=parent["span_id"],
                        correlation_id=correlation,
                    )
                )
    round13 = []
    for sequence, generation in enumerate(base["receipt"]["generations"]):
        request_id = generation["request_id"]
        spans = []
        for row in base["observer"]["rows"]:
            if row["request_id"] != request_id or row["component"] not in {
                "QSA_INDEXER",
                "GDN",
                "NPU_RESIDENT",
            }:
                continue
            spans.append(
                {
                    "component": row["component"],
                    "layer": row["layer"],
                    "device_ms": row["elapsed_ms"],
                    "device_status": "COMPLETE",
                    "observer_errors": [],
                }
            )
        started = 1_000_000 + sequence * 10_000
        round13.append(
            {
                "schema": 1,
                "status": "ROUND13_P0_BATCH",
                "request_id": request_id,
                "mode": "decode",
                "batch_sequence": 0,
                "started_ns": started,
                "ended_ns": started + 8_000,
                "duration_ns": 8_000,
                "spans": spans,
                "dropped_spans": 0,
                "observer_errors": [],
            }
        )
    return base["receipt"], runtime_rows, round13


class FreshA1EvidenceTests(unittest.TestCase):
    def _bundle(self, block: str):
        receipt, runtime, round13 = _raw_inputs(block)
        observer = normalize_observer_rows(
            receipt,
            runtime,
            round13,
            runtime_bindings=RUNTIME_BINDINGS,
            round13_bindings=ROUND13_BINDINGS,
        )
        return {"receipt": receipt, "observer": observer}

    def test_runtime_and_round13_rows_reach_conservative_two_block_decision(self):
        payload = analysis_payload(self._bundle("A"), self._bundle("B"))

        self.assertEqual(
            payload["critical_path"]["status"], "A1_INSUFFICIENT_ATTRIBUTION"
        )
        self.assertEqual(
            payload["decision"]["status"], "A1_INSUFFICIENT_ATTRIBUTION"
        )
        self.assertIsNone(payload["decision"]["selected_candidate"])
        self.assertEqual(
            payload["decision"]["gate_results"],
            {
                "A2": "NOT_SELECTED_EVIDENCE_GAP",
                "A3": "NOT_SELECTED_EVIDENCE_GAP",
            },
        )
        for block in payload["critical_path"]["blocks"]:
            self.assertEqual(
                block["candidate_regions"]["status"],
                "INSUFFICIENT_ATTRIBUTION",
            )
            self.assertEqual(block["candidate_regions"]["gate_met"], {"A2": False, "A3": False})
            self.assertEqual(
                block["measurement_event_flush"]["status"],
                "COMPLETE_MEASUREMENT_BATCHES_SERIALIZED",
            )
            self.assertFalse(
                block["measurement_event_flush"][
                    "final_canary_device_completion_required"
                ]
            )

    def test_device_event_coverage_is_required_only_for_measurements(self):
        bundles = []
        for block in ("A", "B"):
            receipt, runtime, round13 = _raw_inputs(block)
            measurement_request_ids = {
                row["request_id"]
                for row in receipt["generations"]
                if row["role"] in {"W1_MEASUREMENT", "W2_MEASUREMENT"}
            }
            for batch in round13:
                if batch["request_id"] not in measurement_request_ids:
                    batch["spans"] = []
            observer = normalize_observer_rows(
                receipt,
                runtime,
                round13,
                runtime_bindings=RUNTIME_BINDINGS,
                round13_bindings=ROUND13_BINDINGS,
            )
            self.assertEqual(
                observer["measurement_event_flush"]["request_ids"],
                sorted(measurement_request_ids),
            )
            bundles.append({"receipt": receipt, "observer": observer})

        payload = analysis_payload(*bundles)

        self.assertEqual(
            payload["decision"]["status"], "A1_INSUFFICIENT_ATTRIBUTION"
        )
        self.assertIsNone(payload["decision"]["selected_candidate"])

    def test_normalizer_rejects_incomplete_host48_coverage(self):
        receipt, runtime, round13 = _raw_inputs("A")
        runtime = [
            row
            for row in runtime
            if not (
                row["request_id"] == receipt["generations"][0]["request_id"]
                and row["component"] == "HOST_SYNC"
                and row["layer"] == 17
            )
        ]
        with self.assertRaisesRegex(EvidenceNormalizationError, "HOST_SYNC.*0..47"):
            normalize_observer_rows(
                receipt,
                runtime,
                round13,
                runtime_bindings=RUNTIME_BINDINGS,
                round13_bindings=ROUND13_BINDINGS,
            )

    def test_normalizer_requires_serialized_short_drain_as_measurement_flush_witness(self):
        receipt, runtime, round13 = _raw_inputs("A")
        short_drain_id = next(
            row["request_id"]
            for row in receipt["generations"]
            if row["role"] == "SHORT_DRAIN"
        )
        round13 = [
            batch for batch in round13 if batch["request_id"] != short_drain_id
        ]

        with self.assertRaisesRegex(
            EvidenceNormalizationError, "SHORT_DRAIN.*flush witness"
        ):
            normalize_observer_rows(
                receipt,
                runtime,
                round13,
                runtime_bindings=RUNTIME_BINDINGS,
                round13_bindings=ROUND13_BINDINGS,
            )


if __name__ == "__main__":
    unittest.main()
