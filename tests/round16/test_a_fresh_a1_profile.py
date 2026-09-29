from __future__ import annotations

import json
from pathlib import Path
import unittest

from round16.a.fresh_a1_profile import (
    BLOCK_IDS,
    DEFAULT_ENABLED,
    BudgetExceededError,
    BudgetLedger,
    FreshA1Observer,
    SpanIdentity,
    block_contract,
    controller_plan,
    cpu_expert_unknown,
    device_elapsed_record,
    manifest_payload,
    spend_generation,
)


class FreshA1ProfileTests(unittest.TestCase):
    def test_frozen_contract_artifact_matches_runtime_payload(self) -> None:
        root = Path(__file__).resolve().parents[2]
        artifact = json.loads(
            (root / "artifacts/round16-a/FRESH_A1_PROFILE_CONTRACT.json").read_text()
        )

        self.assertEqual(artifact, manifest_payload())

    def test_manifest_has_two_new_immutable_blocks_and_twelve_generation_budget(self) -> None:
        payload = manifest_payload()

        self.assertFalse(DEFAULT_ENABLED)
        self.assertFalse(payload["default_enabled"])
        self.assertEqual(payload["stage"], "ROUND16_A1_FRESH_PROFILE")
        self.assertEqual(payload["source_basis"]["parent"], "ROUND14A_V295_PROMOTED_GDN_S0")
        self.assertEqual(
            payload["source_basis"]["round15_terminal_status"],
            "ROUND15_COMPLETE_SIMPLIFIED_DIRECTIONAL_NO_PROMOTION",
        )
        self.assertEqual(
            payload["source_basis"]["observer_sha256"],
            "a4bd6c90bc05ec09e51d14194fb2eaf38e00dcb5578d48b91bfdfc1f4b392cc0",
        )
        self.assertEqual(
            payload["source_basis"]["scheduler_sha256"],
            "5b96f89ee24f49671b3c35960922d7f93b2133c62e087642d6bdde3ea20b1c6b",
        )
        self.assertEqual(
            payload["source_basis"]["launcher_sha256"],
            "daad3cf0ad99c870c4a884ca0440c4f88d5c887984c4735dff2fe72b5c291f1c",
        )
        self.assertEqual(payload["snapshot_builder"]["mutation_scope"], "COPIED_TREE_ONLY")
        self.assertEqual(payload["snapshot_builder"]["parent_write_count"], 0)
        self.assertEqual(
            payload["snapshot_builder"]["actual_boundaries"],
            [
                "KTEPWrapperMethod.apply Host parent",
                "KTEPWrapperMethod._submit_cpu_forward",
                "KTEPWrapperMethod._sync_cpu_forward",
                "KTEPWrapperMethod.apply post-wait_event merge",
                "round13_attribution QSA_INDEXER/GDN/NPU_RESIDENT direct samples",
            ],
        )
        self.assertEqual([row["block"] for row in payload["blocks"]], ["A", "B"])
        self.assertEqual(
            [row["block_id"] for row in payload["blocks"]],
            [BLOCK_IDS["A"], BLOCK_IDS["B"]],
        )
        self.assertEqual(len(set(BLOCK_IDS.values())), 2)
        self.assertEqual(payload["budget"]["generation_attempts_total"], 12)
        self.assertEqual(payload["budget"]["generation_attempts_per_block"], 6)
        self.assertEqual(payload["budget"]["automatic_retries"], 0)
        self.assertEqual(
            [row["role"] for row in payload["schedule"]],
            [
                "W1_WARMUP",
                "W2_WARMUP",
                "W1_MEASUREMENT",
                "W2_MEASUREMENT",
                "SHORT_DRAIN",
                "FIXED_CANARY",
            ],
        )
        drain, canary = payload["schedule"][4:]
        self.assertTrue(drain["ignore_eos"])
        self.assertFalse(canary["ignore_eos"])
        self.assertEqual(canary["max_new_tokens"], 32)
        self.assertEqual(canary["validator"], {"kind": "exact_text", "expected": "391"})
        self.assertEqual(
            canary["input_ids_sha256"],
            "4d2584778859d8b34fa388d6fb09925c1da1a0197945de1298f09d199dce489d",
        )
        json.dumps(payload, allow_nan=False)

    def test_snapshot_and_controller_wiring_are_default_off_and_non_executable(self) -> None:
        contract = block_contract("A")
        plan = controller_plan("A")

        self.assertEqual(contract.block_id, BLOCK_IDS["A"])
        self.assertEqual(contract.generation_limit, 6)
        self.assertFalse(contract.enabled)
        self.assertFalse(plan["enabled"])
        self.assertFalse(plan["executable"])
        self.assertEqual(plan["environment"]["ROUND16_A1_PROFILE_ENABLE"], "0")
        self.assertEqual(plan["snapshot"]["block_id"], BLOCK_IDS["A"])
        self.assertNotIn("observer_hooks", plan)
        self.assertEqual(
            plan["observer_boundary_catalog"],
            {
                "parent": "KTEPWrapperMethod.apply",
                "host_dispatch": "KTEPWrapperMethod._submit_cpu_forward",
                "host_sync": "KTEPWrapperMethod._sync_cpu_forward",
                "npu_shared": "SharedFullContext.gpu_method.apply",
                "npu_resident": "KTEPWrapperMethod.gpu_method.apply",
                "host_merge": "KTEPWrapperMethod.apply after main-stream wait_event",
                "scheduler_before": "round13_attribution.before",
                "scheduler_after": "round13_attribution.after",
            },
        )
        self.assertEqual(
            plan["observer_installation"],
            {
                "round16_a1_host_layers": list(range(48)),
                "round13_direct_device_components": [
                    "QSA_INDEXER",
                    "GDN",
                    "NPU_RESIDENT",
                ],
                "round13_device_sample_layers": [0, 24, 47],
                "a1_npu_shared_installed": False,
                "catalog_is_installation_claim": False,
            },
        )

    def test_budget_ledger_is_immutable_and_enforces_per_block_and_total_limits(self) -> None:
        empty = BudgetLedger.empty()
        ledger = empty
        for block in ("A", "B"):
            for ordinal in range(1, 7):
                ledger = spend_generation(
                    ledger,
                    block_id=BLOCK_IDS[block],
                    request_id=f"{block.lower()}-{ordinal}",
                )

        self.assertEqual(empty.total_used, 0)
        self.assertEqual(ledger.total_used, 12)
        self.assertEqual(ledger.used_for(BLOCK_IDS["A"]), 6)
        self.assertEqual(ledger.used_for(BLOCK_IDS["B"]), 6)

        with self.assertRaisesRegex(BudgetExceededError, "duplicate"):
            spend_generation(
                ledger,
                block_id=BLOCK_IDS["A"],
                request_id="a-1",
            )
        with self.assertRaisesRegex(BudgetExceededError, "block"):
            spend_generation(
                ledger,
                block_id=BLOCK_IDS["A"],
                request_id="a-7",
            )
        with self.assertRaisesRegex(BudgetExceededError, "immutable block"):
            spend_generation(ledger, block_id="legacy-v338", request_id="x")

    def test_default_off_observer_does_not_read_clock_or_emit_rows(self) -> None:
        def forbidden_clock() -> int:
            raise AssertionError("default-off observer touched the clock")

        observer = FreshA1Observer(clock=forbidden_clock)
        identity = SpanIdentity(
            component="HOST_DISPATCH",
            span_id="span-1",
            parent_span_id="batch-1",
            stream_id="host-main",
            start_event_id="event-1-start",
            end_event_id="event-1-end",
            correlation_id="cpu-task-1",
        )

        result = observer.observe_call(identity, lambda value: value + 1, 4)

        self.assertEqual(result, 5)
        self.assertEqual(observer.rows, ())

    def test_enabled_host_observer_emits_self_describing_identity(self) -> None:
        ticks = iter((100, 175))
        observer = FreshA1Observer(enabled=True, clock=lambda: next(ticks))
        identity = SpanIdentity(
            component="HOST_SYNC",
            span_id="span-2",
            parent_span_id="parent-2",
            stream_id="host-main",
            start_event_id="event-2-start",
            end_event_id="event-2-end",
            correlation_id="cpu-task-2",
        )

        self.assertEqual(observer.observe_call(identity, lambda: "value"), "value")
        row = observer.rows[0]

        self.assertEqual(row["clock_domain"], "HOST_MONOTONIC_NS")
        self.assertEqual(row["start_ns"], 100)
        self.assertEqual(row["end_ns"], 175)
        self.assertEqual(row["duration_ns"], 75)
        self.assertEqual(row["span_id"], "span-2")
        self.assertEqual(row["parent_span_id"], "parent-2")
        self.assertEqual(row["stream_id"], "host-main")
        self.assertEqual(row["start_event_id"], "event-2-start")
        self.assertEqual(row["end_event_id"], "event-2-end")
        self.assertEqual(row["correlation_id"], "cpu-task-2")
        self.assertEqual(row["outcome"], "return")

    def test_observer_preserves_exception_identity_and_records_error_outcome(self) -> None:
        ticks = iter((8, 13))
        observer = FreshA1Observer(enabled=True, clock=lambda: next(ticks))
        identity = SpanIdentity(
            component="HOST_DISPATCH",
            span_id="span-error",
            parent_span_id="batch-error",
            stream_id="host-main",
            start_event_id="event-error-start",
            end_event_id="event-error-end",
            correlation_id="cpu-task-error",
        )
        expected = LookupError("same object")

        def fail() -> None:
            raise expected

        with self.assertRaises(LookupError) as raised:
            observer.observe_call(identity, fail)

        self.assertIs(raised.exception, expected)
        self.assertEqual(observer.rows[0]["outcome"], "error")
        self.assertEqual(observer.rows[0]["duration_ns"], 5)

    def test_device_record_and_cpu_expert_unknown_never_fabricate_cross_clock_time(self) -> None:
        identity = SpanIdentity(
            component="NPU",
            span_id="span-device",
            parent_span_id="parent-device",
            stream_id="ascend-current-stream:0",
            start_event_id="npu-event-start",
            end_event_id="npu-event-end",
            correlation_id="hybrid-task-7",
        )

        device = device_elapsed_record(identity, elapsed_ms=1.25)
        unknown = cpu_expert_unknown(identity.with_component("CPU_EXPERT"))

        self.assertEqual(device["clock_domain"], "NPU_EVENT_ELAPSED_MS")
        self.assertEqual(device["elapsed_ms"], 1.25)
        self.assertIsNone(device["start_ns"])
        self.assertIsNone(device["end_ns"])
        self.assertEqual(unknown["status"], "UNKNOWN_NATIVE_ABI_NO_CALLBACK")
        self.assertEqual(unknown["clock_domain"], "NATIVE_WORKER_CLOCK_UNAVAILABLE")
        self.assertIsNone(unknown["start_ns"])
        self.assertIsNone(unknown["end_ns"])
        self.assertIsNone(unknown["duration_ns"])
        self.assertEqual(unknown["correlation_id"], "hybrid-task-7")
        self.assertEqual(unknown["derivation"], "NOT_INFERRED_FROM_HOST_BOOKENDS")


if __name__ == "__main__":
    unittest.main()
