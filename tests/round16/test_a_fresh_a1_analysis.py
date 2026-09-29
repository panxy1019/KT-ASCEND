from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from round16.a.fresh_a1_analysis import (
    EvidenceContractError,
    analysis_payload,
    dependency_dag_payload,
    load_evidence_bundle,
    render_critical_path_markdown,
    write_analysis_outputs,
)
from round16.a.fresh_a1_profile import BLOCK_IDS, block_contract, manifest_payload
from round16.a.fresh_a1_snapshot import canonical_json_sha256


QSA_LAYERS = (3, 23, 47)
GDN_LAYERS = (0, 24, 46)
EXPERT_LAYERS = (0, 24, 47)


def _observed_request(block_id: str, schedule: dict, request_id: str) -> dict:
    return {
        "block_id": block_id,
        "request_id": request_id,
        "role": schedule["role"],
        "input_tokens": schedule["input_tokens"],
        "max_new_tokens": (
            schedule["max_new_tokens"]
            if "max_new_tokens" in schedule
            else schedule["output_tokens"]
        ),
        "ignore_eos": schedule.get("ignore_eos", False),
        "fixture_sha256": schedule.get("fixture_sha256"),
        "input_ids_sha256": schedule.get("input_ids_sha256"),
        "messages_sha256": schedule.get("messages_sha256"),
    }


def _identity(block_id: str) -> dict:
    return {
        "model_revision": "de4b8e4d43b917e7706784d8bb445c9af86a3540",
        "fixture_sha256": "21e4f82531a58881f2fa251fda50ef3f9909c589336164b77fe732a77abdca95",
        "observer_contract": "round16-a1-observer-v001",
        "provider_state": "OFF",
        "schedule_owner": "round16-contract-66",
    }


def _event(
    *,
    block_id: str,
    request_id: str,
    component: str,
    layer: int,
    serial: int,
    clock_domain: str,
    value: float | int | None,
    parent_span_id: str,
    correlation_id: str,
) -> dict:
    stem = f"{block_id}:{request_id}:{component}:{layer}:{serial}"
    row = {
        "schema": 1,
        "block_id": block_id,
        "request_id": request_id,
        "component": component,
        "layer": layer,
        "span_id": stem,
        "parent_span_id": parent_span_id,
        "stream_id": "host-main" if clock_domain == "HOST_MONOTONIC_NS" else "npu:0",
        "start_event_id": f"{stem}:start",
        "end_event_id": f"{stem}:end",
        "correlation_id": correlation_id,
        "clock_domain": clock_domain,
    }
    if clock_domain == "HOST_MONOTONIC_NS":
        row.update(
            {
                "start_ns": 100,
                "end_ns": 100 + int(value),
                "duration_ns": int(value),
                "elapsed_ms": None,
                "status": "COMPLETE",
                "derivation": "DIRECT_OBSERVATION",
            }
        )
    elif clock_domain == "NPU_EVENT_ELAPSED_MS":
        row.update(
            {
                "start_ns": None,
                "end_ns": None,
                "duration_ns": None,
                "elapsed_ms": float(value),
                "status": "COMPLETE",
                "query_policy": "DEFERRED_QUERY_NO_SYNCHRONIZE",
                "derivation": "DIRECT_OBSERVATION",
            }
        )
    else:
        row.update(
            {
                "start_ns": None,
                "end_ns": None,
                "duration_ns": None,
                "elapsed_ms": None,
                "status": "UNKNOWN_NATIVE_ABI_NO_CALLBACK",
                "derivation": "NOT_INFERRED_FROM_HOST_BOOKENDS",
            }
        )
    return row


def _block(
    block: str,
    *,
    gdn_ms: float,
    expert_ms: float,
    a2_fraction: float | None = None,
    a3_fraction: float | None = None,
    a2_window_available: bool = True,
    a3_outside_resident_graph: bool = True,
) -> dict:
    block_id = BLOCK_IDS[block]
    contract = block_contract(block)
    schedule = manifest_payload()["schedule"]
    schedule_sha = canonical_json_sha256(schedule)
    source_identity = _identity(block_id)
    generations = []
    rows = []
    serial = 0
    for item in schedule:
        request_id = f"{block_id}:{item['ordinal']}:{item['role']}"
        generations.append(
            {
                "ordinal": item["ordinal"],
                "role": item["role"],
                "request_id": request_id,
                "status": "COMPLETE",
                "schedule": copy.deepcopy(item),
                "driver": {
                    "status": "COMPLETE",
                    "observed_request": _observed_request(block_id, item, request_id),
                    "source_identity": copy.deepcopy(source_identity),
                },
            }
        )
        for layer in QSA_LAYERS:
            serial += 1
            rows.append(
                _event(
                    block_id=block_id,
                    request_id=request_id,
                    component="QSA_INDEXER",
                    layer=layer,
                    serial=serial,
                    clock_domain="NPU_EVENT_ELAPSED_MS",
                    value=2.0,
                    parent_span_id=f"{request_id}:batch",
                    correlation_id=f"{request_id}:qsa:{layer}",
                )
            )
        for layer in GDN_LAYERS:
            serial += 1
            rows.append(
                _event(
                    block_id=block_id,
                    request_id=request_id,
                    component="GDN",
                    layer=layer,
                    serial=serial,
                    clock_domain="NPU_EVENT_ELAPSED_MS",
                    value=gdn_ms,
                    parent_span_id=f"{request_id}:batch",
                    correlation_id=f"{request_id}:gdn:{layer}",
                )
            )
        for layer in EXPERT_LAYERS:
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
            rows.append(parent)
            for component, domain, value in (
                ("HOST_DISPATCH", "HOST_MONOTONIC_NS", 100),
                ("NPU_RESIDENT", "NPU_EVENT_ELAPSED_MS", expert_ms),
                ("HOST_SYNC", "HOST_MONOTONIC_NS", 200),
                ("HOST_MERGE", "HOST_MONOTONIC_NS", 50),
                ("CPU_EXPERT", "NATIVE_WORKER_CLOCK_UNAVAILABLE", None),
            ):
                serial += 1
                rows.append(
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
    candidate_regions = []
    if a2_fraction is not None and a3_fraction is not None:
        decode_phase_wall_ns = 100_000
        measurement_request_ids = {
            row["request_id"]
            for row in generations
            if row["role"] in ("W1_MEASUREMENT", "W2_MEASUREMENT")
        }
        for request_id in sorted(measurement_request_ids):
            serial += 1
            decode_wall = _event(
                block_id=block_id,
                request_id=request_id,
                component="DECODE_PHASE_WALL",
                layer=-1,
                serial=serial,
                clock_domain="HOST_MONOTONIC_NS",
                value=decode_phase_wall_ns,
                parent_span_id=f"{request_id}:batch",
                correlation_id=f"{request_id}:decode-phase",
            )
            rows.append(decode_wall)
            host_sync_spans = [
                row["span_id"]
                for row in rows
                if row["request_id"] == request_id
                and row["component"] == "HOST_SYNC"
            ]
            host_launch_spans = [
                row["span_id"]
                for row in rows
                if row["request_id"] == request_id
                and row["component"] == "HOST_DISPATCH"
            ]
            npu_window_spans = [
                row["span_id"]
                for row in rows
                if row["request_id"] == request_id
                and row["component"] in ("NPU_RESIDENT", "NPU_SHARED")
            ]
            candidate_regions.extend(
                [
                    {
                        "region_id": f"{request_id}:A2:host-sync-exposed-wait",
                        "candidate": "A2",
                        "component": "HOST_SYNC_EXPOSED_WAIT",
                        "request_id": request_id,
                        "clock_domain": "HOST_MONOTONIC_NS",
                        "phase": "DECODE",
                        "duration_ns": int(decode_phase_wall_ns * a2_fraction),
                        "start_ns": 1_000,
                        "end_ns": 1_000
                        + int(decode_phase_wall_ns * a2_fraction),
                        "decode_phase_wall_ns": decode_phase_wall_ns,
                        "decode_phase_wall_span_id": decode_wall["span_id"],
                        "derivation": "DIRECT_CRITICAL_PATH_INTERVAL_NO_SUM",
                        "source_span_ids": host_sync_spans,
                        "dependency_proof": {
                            "type": "SAME_LAYER_INDEPENDENT_NPU_WINDOW",
                            "status": (
                                "AVAILABLE_UNUSED"
                                if a2_window_available
                                else "NOT_PROVEN"
                            ),
                            "layers": list(EXPERT_LAYERS),
                            "source_span_ids": npu_window_spans,
                            "target": "CPU_EXPERT",
                            "clock_relation": "DEPENDENCY_ONLY_NO_TIME_ARITHMETIC",
                        },
                    },
                    {
                        "region_id": f"{request_id}:A3:host-launch-metadata",
                        "candidate": "A3",
                        "component": "HOST_LAUNCH_METADATA_CRITICAL",
                        "request_id": request_id,
                        "clock_domain": "HOST_MONOTONIC_NS",
                        "phase": "DECODE",
                        "duration_ns": int(decode_phase_wall_ns * a3_fraction),
                        "start_ns": 2_000,
                        "end_ns": 2_000
                        + int(decode_phase_wall_ns * a3_fraction),
                        "decode_phase_wall_ns": decode_phase_wall_ns,
                        "decode_phase_wall_span_id": decode_wall["span_id"],
                        "derivation": "DIRECT_CRITICAL_PATH_INTERVAL_NO_SUM",
                        "source_span_ids": host_launch_spans,
                        "outside_resident_only_graph": a3_outside_resident_graph,
                    },
                ]
            )
    return {
        "receipt": {
            "schema": 1,
            "stage": "ROUND16_A1_FRESH_PROFILE",
            "status": "RAW_DEVICE_RESULT_UNREVIEWED",
            "block_id": block_id,
            "snapshot_id": contract.snapshot_id,
            "source_manifest_sha256": ("a" if block == "A" else "b") * 64,
            "schedule_sha256": schedule_sha,
            "generation_attempts": 6,
            "generations": generations,
            "candidate_selected": None,
        },
        "observer": {
            "schema": 1,
            "synchronize_inserted": False,
            "query_policy": "DEFERRED_QUERY_NO_SYNCHRONIZE",
            "source_identity": source_identity,
            "rows": rows,
            "aggregations": [],
            "candidate_regions": candidate_regions,
        },
    }


class FreshA1AnalysisTests(unittest.TestCase):
    def test_pre_device_templates_are_preserved_and_final_artifacts_use_fresh_blocks(self) -> None:
        root = Path(__file__).resolve().parents[2]
        payload = analysis_payload()

        self.assertEqual(payload["critical_path"]["status"], "DEVICE_EVIDENCE_PENDING")
        self.assertEqual(payload["decision"]["status"], "DEVICE_EVIDENCE_PENDING")
        self.assertIsNone(payload["decision"]["selected_candidate"])
        self.assertFalse(payload["decision"]["historical_evidence_used"])
        self.assertEqual(
            json.loads(
                (
                    root
                    / "artifacts/round16-a/A1_CRITICAL_PATH_PRE_DEVICE.json"
                ).read_text()
            ),
            payload["critical_path"],
        )
        self.assertEqual(
            json.loads(
                (
                    root
                    / "artifacts/round16-a/A_REGION_DEPENDENCY_DAG_PRE_DEVICE.json"
                ).read_text()
            ),
            payload["dag"],
        )
        self.assertEqual(
            json.loads(
                (
                    root
                    / "artifacts/round16-a/A_CANDIDATE_DECISION_PRE_DEVICE.json"
                ).read_text()
            ),
            payload["decision"],
        )

        analysis_dir = (
            root
            / "artifacts/round16-a/fresh-profile-b-v003/round16-a1-analysis"
        )
        for name in (
            "A1_CRITICAL_PATH.json",
            "A_REGION_DEPENDENCY_DAG.json",
            "A_CANDIDATE_DECISION.json",
        ):
            final = json.loads((root / "artifacts/round16-a" / name).read_text())
            fresh = json.loads((analysis_dir / name).read_text())
            self.assertEqual(final, fresh)

        final_decision = json.loads(
            (root / "artifacts/round16-a/A_CANDIDATE_DECISION.json").read_text()
        )
        self.assertEqual(final_decision["status"], "A1_INSUFFICIENT_ATTRIBUTION")
        self.assertEqual(
            final_decision["fresh_blocks_present"],
            [
                "round16-a1-fresh-profile-a-v001",
                "round16-a1-fresh-profile-b-v001",
            ],
        )
        self.assertIsNone(final_decision["selected_candidate"])

    def test_dependency_dag_freezes_non_additive_parent_and_separate_clock_domains(self) -> None:
        dag = dependency_dag_payload()

        self.assertEqual(dag["status"], "FROZEN_OFFLINE")
        self.assertEqual(
            dag["arithmetic_policy"]["operation"],
            "DIRECT_REGION_FRACTION_OF_DECODE_WALL",
        )
        self.assertFalse(dag["arithmetic_policy"]["parent_child_sum_allowed"])
        self.assertFalse(dag["arithmetic_policy"]["cross_clock_sum_allowed"])
        self.assertEqual(
            dag["evidence_input_contract"]["loader"],
            "load_evidence_bundle",
        )
        self.assertEqual(
            dag["evidence_input_contract"]["identity_binding"],
            ["receipt_sha256", "rows_sha256", "observer_path"],
        )
        self.assertIn(
            {
                "from": "PARENT_ROUTED_EXPERT",
                "to": "NPU_EXPERT",
                "relation": "CONTAINS_NON_ADDITIVE",
            },
            dag["edges"],
        )
        cpu = next(node for node in dag["nodes"] if node["id"] == "CPU_EXPERT")
        self.assertEqual(cpu["status"], "UNKNOWN_NATIVE_ABI_NO_CALLBACK")
        self.assertIsNone(cpu["metric"])

        a2 = dag["candidate_gates"]["A2"]
        self.assertEqual(a2["numerator"], "HOST_SYNC_EXPOSED_WAIT")
        self.assertEqual(a2["clock_domain"], "HOST_MONOTONIC_NS")
        self.assertEqual(a2["minimum_decode_wall_fraction"], 0.10)
        self.assertTrue(a2["requires_same_layer_independent_npu_window"])
        self.assertNotIn("GDN", a2["numerator"])
        a3 = dag["candidate_gates"]["A3"]
        self.assertEqual(a3["numerator"], "HOST_LAUNCH_METADATA_CRITICAL")
        self.assertTrue(a3["requires_outside_resident_only_graph"])
        self.assertTrue(a2["requires_primary_actionable_region_in_each_block"])
        self.assertTrue(a3["requires_primary_actionable_region_in_each_block"])

    def test_two_fresh_blocks_select_a2_from_exposed_wait_and_independent_window(self) -> None:
        payload = analysis_payload(
            _block(
                "A", gdn_ms=2.0, expert_ms=20.0, a2_fraction=0.12, a3_fraction=0.04
            ),
            _block(
                "B", gdn_ms=20.0, expert_ms=2.0, a2_fraction=0.11, a3_fraction=0.03
            ),
        )

        self.assertEqual(payload["critical_path"]["status"], "PASS_TWO_FRESH_BLOCKS")
        self.assertEqual(payload["decision"]["selected_candidate"], "A2")
        self.assertEqual(payload["decision"]["status"], "SELECT_A2")
        self.assertEqual(
            payload["critical_path"]["arithmetic"]["operation"],
            "DIRECT_REGION_FRACTION_OF_DECODE_WALL",
        )
        self.assertEqual(payload["critical_path"]["arithmetic"]["added_terms"], [])
        self.assertEqual(
            payload["critical_path"]["unknown_regions"][0]["status"],
            "UNKNOWN_NATIVE_ABI_NO_CALLBACK",
        )
        self.assertIsNone(
            payload["critical_path"]["unknown_regions"][0]["duration"]
        )
        markdown = render_critical_path_markdown(payload["critical_path"])
        self.assertIn("SELECT_A2", markdown)
        self.assertIn("UNKNOWN_NATIVE_ABI_NO_CALLBACK", markdown)

    def test_two_fresh_blocks_select_a3_or_no_candidate_only_when_blocks_agree(self) -> None:
        a3 = analysis_payload(
            _block(
                "A", gdn_ms=30.0, expert_ms=1.0, a2_fraction=0.03, a3_fraction=0.12
            ),
            _block(
                "B", gdn_ms=1.0, expert_ms=30.0, a2_fraction=0.04, a3_fraction=0.11
            ),
        )
        disagreement = analysis_payload(
            _block(
                "A", gdn_ms=10.0, expert_ms=2.0, a2_fraction=0.12, a3_fraction=0.03
            ),
            _block(
                "B", gdn_ms=2.0, expert_ms=10.0, a2_fraction=0.03, a3_fraction=0.12
            ),
        )

        self.assertEqual(a3["decision"]["selected_candidate"], "A3")
        self.assertEqual(a3["decision"]["status"], "SELECT_A3")
        self.assertEqual(disagreement["decision"]["selected_candidate"], "NO_CANDIDATE")
        self.assertEqual(disagreement["decision"]["status"], "NO_CANDIDATE")

    def test_kernel_durations_cannot_substitute_for_a2_or_a3_gate_evidence(self) -> None:
        payload = analysis_payload(
            _block("A", gdn_ms=100.0, expert_ms=1.0),
            _block("B", gdn_ms=1.0, expert_ms=100.0),
        )

        self.assertEqual(payload["critical_path"]["status"], "DEVICE_EVIDENCE_PENDING")
        self.assertEqual(payload["decision"]["status"], "DEVICE_EVIDENCE_PENDING")
        self.assertIsNone(payload["decision"]["selected_candidate"])
        self.assertEqual(
            payload["decision"]["missing_candidate_evidence"],
            ["A2_HOST_SYNC_EXPOSURE", "A3_HOST_LAUNCH_METADATA"],
        )

    def test_a2_requires_independent_window_and_a3_requires_region_beyond_resident_graph(self) -> None:
        a2_unproven = analysis_payload(
            _block(
                "A",
                gdn_ms=1.0,
                expert_ms=1.0,
                a2_fraction=0.12,
                a3_fraction=0.01,
                a2_window_available=False,
            ),
            _block(
                "B",
                gdn_ms=1.0,
                expert_ms=1.0,
                a2_fraction=0.12,
                a3_fraction=0.01,
                a2_window_available=False,
            ),
        )
        a3_in_old_graph = analysis_payload(
            _block(
                "A",
                gdn_ms=1.0,
                expert_ms=1.0,
                a2_fraction=0.01,
                a3_fraction=0.12,
                a3_outside_resident_graph=False,
            ),
            _block(
                "B",
                gdn_ms=1.0,
                expert_ms=1.0,
                a2_fraction=0.01,
                a3_fraction=0.12,
                a3_outside_resident_graph=False,
            ),
        )

        self.assertEqual(a2_unproven["decision"]["status"], "NO_CANDIDATE")
        self.assertEqual(
            a2_unproven["decision"]["gate_results"]["A2"], "NO_CURRENT_A2"
        )
        self.assertEqual(a3_in_old_graph["decision"]["status"], "NO_CANDIDATE")
        self.assertEqual(
            a3_in_old_graph["decision"]["gate_results"]["A3"], "NO_CURRENT_A3"
        )

    def test_candidate_must_be_primary_in_each_block_and_reference_direct_decode_wall(self) -> None:
        dominated = analysis_payload(
            _block(
                "A", gdn_ms=1.0, expert_ms=1.0, a2_fraction=0.12, a3_fraction=0.13
            ),
            _block(
                "B", gdn_ms=1.0, expert_ms=1.0, a2_fraction=0.12, a3_fraction=0.03
            ),
        )
        missing_wall = _block(
            "A", gdn_ms=1.0, expert_ms=1.0, a2_fraction=0.12, a3_fraction=0.03
        )
        missing_wall["observer"]["rows"] = [
            row
            for row in missing_wall["observer"]["rows"]
            if row["component"] != "DECODE_PHASE_WALL"
        ]
        good_b = _block(
            "B", gdn_ms=1.0, expert_ms=1.0, a2_fraction=0.12, a3_fraction=0.03
        )

        self.assertEqual(dominated["decision"]["status"], "NO_CANDIDATE")
        self.assertEqual(
            dominated["decision"]["gate_results"]["A2"], "NO_CURRENT_A2"
        )
        with self.assertRaisesRegex(EvidenceContractError, "Decode phase wall"):
            analysis_payload(missing_wall, good_b)

    def test_one_fresh_block_is_pending_and_never_uses_it_as_historical_substitute(self) -> None:
        payload = analysis_payload(_block("A", gdn_ms=10.0, expert_ms=2.0), None)

        self.assertEqual(payload["decision"]["status"], "DEVICE_EVIDENCE_PENDING")
        self.assertIsNone(payload["decision"]["selected_candidate"])
        self.assertEqual(payload["decision"]["fresh_blocks_present"], [BLOCK_IDS["A"]])
        self.assertFalse(payload["decision"]["historical_evidence_used"])

    def test_rejects_missing_request_layer_or_event(self) -> None:
        good_a = _block("A", gdn_ms=10.0, expert_ms=2.0)
        good_b = _block("B", gdn_ms=10.0, expert_ms=2.0)
        cases = []
        missing_request = copy.deepcopy(good_a)
        missing_request["receipt"]["generations"].pop()
        cases.append((missing_request, "six requests"))
        missing_observer_request = copy.deepcopy(good_a)
        absent_request_id = missing_observer_request["receipt"]["generations"][0][
            "request_id"
        ]
        missing_observer_request["observer"]["rows"] = [
            row
            for row in missing_observer_request["observer"]["rows"]
            if row["request_id"] != absent_request_id
        ]
        cases.append((missing_observer_request, "observer events missing"))
        missing_layer = copy.deepcopy(good_a)
        missing_layer["observer"]["rows"] = [
            row
            for row in missing_layer["observer"]["rows"]
            if not (row["component"] == "GDN" and row["layer"] == 46)
        ]
        cases.append((missing_layer, "GDN.*layer 46"))
        missing_event = copy.deepcopy(good_a)
        del missing_event["observer"]["rows"][0]["end_event_id"]
        cases.append((missing_event, "event identity"))

        for bad, pattern in cases:
            with self.subTest(pattern=pattern):
                with self.assertRaisesRegex(EvidenceContractError, pattern):
                    analysis_payload(bad, good_b)

    def test_rejects_duplicate_span_or_event_identity(self) -> None:
        good_a = _block("A", gdn_ms=10.0, expert_ms=2.0)
        good_b = _block("B", gdn_ms=10.0, expert_ms=2.0)
        for field in ("span_id", "start_event_id"):
            bad = copy.deepcopy(good_a)
            bad["observer"]["rows"][1][field] = bad["observer"]["rows"][0][field]
            with self.subTest(field=field):
                with self.assertRaisesRegex(EvidenceContractError, "duplicate"):
                    analysis_payload(bad, good_b)

    def test_rejects_nested_duplicate_or_cross_clock_aggregation(self) -> None:
        good_a = _block("A", gdn_ms=10.0, expert_ms=2.0)
        good_b = _block("B", gdn_ms=10.0, expert_ms=2.0)
        parent = next(
            row
            for row in good_a["observer"]["rows"]
            if row["component"] == "PARENT_ROUTED_EXPERT"
        )
        child = next(
            row
            for row in good_a["observer"]["rows"]
            if row["parent_span_id"] == parent["span_id"]
        )
        gdn = next(
            row for row in good_a["observer"]["rows"] if row["component"] == "GDN"
        )
        groups = (
            ([parent["span_id"], child["span_id"]], "nested"),
            ([parent["span_id"], gdn["span_id"]], "cross-clock"),
            ([gdn["span_id"], gdn["span_id"]], "duplicate"),
        )
        for terms, pattern in groups:
            bad = copy.deepcopy(good_a)
            bad["observer"]["aggregations"] = [{"operation": "SUM", "terms": terms}]
            with self.subTest(pattern=pattern):
                with self.assertRaisesRegex(EvidenceContractError, pattern):
                    analysis_payload(bad, good_b)

    def test_rejects_observer_synchronization_token_drift_or_source_identity_drift(self) -> None:
        good_a = _block("A", gdn_ms=10.0, expert_ms=2.0)
        good_b = _block("B", gdn_ms=10.0, expert_ms=2.0)
        sync = copy.deepcopy(good_a)
        sync["observer"]["synchronize_inserted"] = True
        token = copy.deepcopy(good_a)
        token["receipt"]["generations"][0]["driver"]["observed_request"][
            "input_tokens"
        ] += 1
        identity = copy.deepcopy(good_b)
        identity["receipt"]["generations"][0]["driver"]["source_identity"][
            "model_revision"
        ] = "drift"
        cpu_zero = copy.deepcopy(good_a)
        cpu_row = next(
            row
            for row in cpu_zero["observer"]["rows"]
            if row["component"] == "CPU_EXPERT"
        )
        cpu_row["duration_ns"] = 0
        for bad_a, bad_b, pattern in (
            (sync, good_b, "synchronization"),
            (token, good_b, "token.*identity"),
            (good_a, identity, "source identity"),
            (cpu_zero, good_b, "cannot be estimated as zero"),
        ):
            with self.subTest(pattern=pattern):
                with self.assertRaisesRegex(EvidenceContractError, pattern):
                    analysis_payload(bad_a, bad_b)

    def test_writer_generates_all_four_review_files(self) -> None:
        with tempfile.TemporaryDirectory() as item:
            paths = write_analysis_outputs(Path(item))

            self.assertEqual(
                set(paths),
                {
                    "critical_path_json",
                    "critical_path_markdown",
                    "dependency_dag_json",
                    "candidate_decision_json",
                },
            )
            for path in paths.values():
                self.assertTrue(path.is_file())
            self.assertIn(
                "DEVICE_EVIDENCE_PENDING",
                paths["critical_path_markdown"].read_text(),
            )

    def test_loader_binds_raw_receipt_metadata_and_observer_rows_by_sha(self) -> None:
        bundle = _block("A", gdn_ms=10.0, expert_ms=2.0)
        with tempfile.TemporaryDirectory() as item:
            root = Path(item)
            receipt_path = root / "raw.json"
            rows_path = root / "observer.jsonl"
            metadata_path = root / "observer-metadata.json"
            bundle["receipt"]["observer_path"] = str(rows_path)
            receipt_path.write_text(json.dumps(bundle["receipt"]) + "\n")
            rows_path.write_text(
                "".join(json.dumps(row) + "\n" for row in bundle["observer"]["rows"])
            )
            metadata = {
                key: value
                for key, value in bundle["observer"].items()
                if key != "rows"
            }
            metadata.update(
                {
                    "receipt_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
                    "rows_path": str(rows_path),
                    "rows_sha256": hashlib.sha256(rows_path.read_bytes()).hexdigest(),
                }
            )
            metadata_path.write_text(json.dumps(metadata) + "\n")

            loaded = load_evidence_bundle(receipt_path, metadata_path)

            self.assertEqual(loaded, bundle)
            metadata["rows_sha256"] = "0" * 64
            metadata_path.write_text(json.dumps(metadata) + "\n")
            with self.assertRaisesRegex(EvidenceContractError, "rows SHA"):
                load_evidence_bundle(receipt_path, metadata_path)


if __name__ == "__main__":
    unittest.main()
