import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "artifacts/round14/compare_round14_b.py"


def load_module():
    spec = importlib.util.spec_from_file_location("compare_round14_b", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def provider(completion_tokens=4):
    return {
        "boundary": {
            "status": "ROUND14_H_BOUNDED_ACCEPT",
            "before_scheduler_dispatch": True,
        },
        "hybrid": {"path": "HYBRID"},
        "qsa": {
            "prefill_strategy": "VECTORIZED",
            "decode_strategy": "DIRECT_BF16_ROW",
            "actual_completion_tokens": completion_tokens,
            "request_drained": True,
        },
        "ple": {"status": "PASS"},
    }


def final(version, block, *, fail_count=2, pending_count=3):
    results = []
    ordinal = 0
    buckets = ("LE128", "129_256", "257_512")
    quality_index = 0
    for bucket in buckets:
        for index in range(16):
            ordinal += 1
            if quality_index < fail_count:
                status, passed = "FAIL_AUTOMATIC", False
            elif quality_index < fail_count + pending_count:
                status, passed = "PENDING_HUMAN", None
            else:
                status, passed = "PASS_AUTOMATIC", True
            quality_index += 1
            completion = 4 + index
            results.append({
                "ordinal": ordinal,
                "kind": "quality",
                "case_id": f"q-{bucket}-{index}",
                "session_id": f"s-{bucket}-{index // 2}",
                "split": "DEV" if index < 4 else "HOLDOUT",
                "bucket": bucket,
                "category": "code" if index % 2 else "chinese",
                "input_tokens": 64 if bucket == "LE128" else (192 if bucket == "129_256" else 384),
                "completion_tokens": completion,
                "quality_status": status,
                "quality_pass": passed,
                "finite_logprobs": True,
                "e2e_seconds": 10.0 + index,
                "provider": provider(completion),
            })
    for bucket in buckets:
        for index in range(4):
            ordinal += 1
            completion = 16
            results.append({
                "ordinal": ordinal,
                "kind": "performance",
                "case_id": f"p-{bucket}-{index}",
                "split": "DEV" if block == "A" else "HOLDOUT",
                "bucket": bucket,
                "category": "performance",
                "input_tokens": 64 if bucket == "LE128" else (192 if bucket == "129_256" else 384),
                "completion_tokens": completion,
                "finite_logprobs": True,
                "finish_reason": "length",
                "e2e_seconds": 20.0 + index,
                "ttft_seconds": 2.0 + index / 10,
                "tpot_seconds": 1.0 + index / 100,
                "provider": provider(completion),
            })
    ordinal += 1
    results.append({
        "ordinal": ordinal, "kind": "boundary_accept", "case_id": "edge512",
        "split": "BOUNDARY", "bucket": "H512", "input_tokens": 512,
        "completion_tokens": 256, "finite_logprobs": True, "finish_reason": "length",
        "e2e_seconds": 200.0, "ttft_seconds": 20.0, "tpot_seconds": 0.7,
        "provider": provider(256),
    })
    ordinal += 1
    results.append({
        "ordinal": ordinal, "kind": "boundary_reject", "case_id": "edge768",
        "split": "BOUNDARY", "bucket": "OUTSIDE", "input_tokens": 768,
        "expected_rejection": True, "http_status": 400,
        "boundary": {"status": "ROUND14_H_BOUNDED_REJECT", "before_scheduler_dispatch": True},
    })
    ordinal += 1
    results.append({
        "ordinal": ordinal, "kind": "cancel", "case_id": "edge512",
        "split": "CONTROL", "bucket": "H512", "cancel_status": "CANCEL_REQUESTED",
        "boundary": {"status": "ROUND14_H_BOUNDED_ACCEPT", "before_scheduler_dispatch": True},
    })
    ordinal += 1
    results.append({
        "ordinal": ordinal, "kind": "reuse", "case_id": "canary_math391",
        "split": "CONTROL", "bucket": "LE128", "completion_tokens": 4,
        "quality_status": "PASS_AUTOMATIC", "quality_pass": True,
        "finite_logprobs": True, "e2e_seconds": 4.0, "provider": provider(4),
    })
    assert ordinal == 64
    return {
        "schema": 1,
        "status": "PASS_ROUND14_B_MODEL_PROCESS",
        "stage": "ROUND14_B_REAL_DISTRIBUTION_AND_BOUNDED_HYBRID",
        "version": version,
        "profile": "H512",
        "block": block,
        "fixture_sha256": "a" * 64,
        "edge_fixture_sha256": "b" * 64,
        "limits": {
            "max_input_tokens": 512,
            "max_output_tokens": 256,
            "max_total_tokens": 768,
            "hybrid_threshold_tokens": 513,
        },
        "profiler": "OFF",
        "generation_budget": 64,
        "generation_attempts": 64,
        "abort_controls": 1,
        "automatic_retries": 0,
        "state_pool_matrix": {"status": "PASS_ACTUAL_CONTEXT_POOL_GEOMETRY"},
        "results": results,
        "cancel_receipts": ["round14-b-cancel-receipt.json"],
        "coverage": {
            "scheduled": {
                "quality": 48, "performance": 12, "boundary_accept": 1,
                "boundary_reject": 1, "cancel": 1, "reuse": 1,
            },
            "automatic_quality_pass": 49 - fail_count,
            "automatic_quality_fail": fail_count,
            "human_review_pending": pending_count,
            "expected_boundary_rejections": 1,
            "unexpected_errors": 0,
        },
        "cleanup": {
            "guard_exit_ack": True,
            "container_removed": True,
            "physical12_after": "EMPTY",
        },
    }


class Round14BComparatorTest(unittest.TestCase):
    def test_two_blocks_with_automatic_failures_retain_h129_default(self):
        api = load_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pairs = []
            for version, block in ((269, "A"), (270, "B")):
                result = root / f"v{version}.json"
                result.write_text(json.dumps(final(version, block)))
                monitor = root / f"v{version}.jsonl"
                monitor.write_text(json.dumps({
                    "hbm_used_mib": 32000,
                    "owned_process_mib": 30000,
                    "host_mem_available_kib": 800000000,
                    "owned_cgroup": {"memory.current": "1000", "memory.peak": "2000"},
                    "numa_meminfo_kib": {
                        node: {"MemFree": 50000000} for node in ("2", "3", "4", "7")
                    },
                }) + "\n")
                pairs.append((result, monitor))
            report = api.compare(pairs)
        self.assertEqual(report["status"], "PASS_ROUND14_B_EVIDENCE_CLOSED")
        self.assertEqual(report["h512_disposition"], "NOT_DEFAULT_QUALITY_REGRESSION")
        self.assertEqual(report["default_profile"], "S0_LEGACY_H129")
        self.assertEqual(report["h768_disposition"], "NOT_RUN_CONDITION_NOT_MET_AND_B_MODEL_GROUP_CAP")
        self.assertEqual(report["coverage"]["quality_requests_per_block"], 48)
        self.assertEqual(report["blocks"]["A"]["quality"]["automatic_fail"], 2)
        self.assertEqual(report["blocks"]["B"]["performance_by_bucket"]["LE128"]["requests"], 4)

    def test_pending_human_review_prevents_default_even_without_auto_fail(self):
        api = load_module()
        documents = [final(269, "A", fail_count=0), final(270, "B", fail_count=0)]
        report = api.compare_documents(documents, resources=[{}, {}])
        self.assertEqual(report["h512_disposition"], "NOT_DEFAULT_PENDING_HUMAN_REVIEW")

    def test_rejects_missing_cleanup_or_wrong_process_identity(self):
        api = load_module()
        a = final(269, "A")
        b = final(270, "B")
        b["cleanup"]["physical12_after"] = "NOT_EMPTY"
        with self.assertRaisesRegex(RuntimeError, "process contract"):
            api.compare_documents([a, b], resources=[{}, {}])

    def test_rejects_missing_boundary_or_provider_contract(self):
        api = load_module()
        a = final(269, "A")
        b = final(270, "B")
        a["results"][60]["provider"]["qsa"]["decode_strategy"] = "OLD"
        with self.assertRaisesRegex(RuntimeError, "provider contract"):
            api.compare_documents([a, b], resources=[{}, {}])

    def test_partial_closure_preserves_failed_block_and_one_complete_block(self):
        api = load_module()
        failed = final(269, "A")
        failed["status"] = "FAIL_ROUND14_B_DRIVER"
        failed["generation_attempts"] = 61
        failed["abort_controls"] = 0
        failed["results"] = failed["results"][:60]
        failed["error"] = "RuntimeError('stream did not produce the exact Decode token budget')"
        failed.pop("cleanup")
        failed.pop("coverage")
        failed["cancel_receipts"] = []
        supervisor = {
            "status": "FAIL", "version": 269, "driver_exit": 1,
            "owned_container_removed": True, "physical12_after": "EMPTY",
            "guard_exit_ack": {"action": "OWNED_EXITED"},
        }
        completed = final(273, "B")
        report = api.close_partial(
            failed, supervisor, completed, resources={"whole_card_hbm_peak_mib": 33000}
        )
        self.assertEqual(report["status"], "ROUND14_B_PARTIAL_WITH_ONE_COMPLETE_BLOCK")
        self.assertEqual(report["complete_blocks"], 1)
        self.assertEqual(report["required_complete_blocks"], 2)
        self.assertEqual(report["default_profile"], "S0_LEGACY_H129")
        self.assertEqual(report["h512_disposition"], "NOT_DEFAULT_QUALITY_REGRESSION")
        self.assertEqual(report["failed_block"]["completed_results"], 60)
        self.assertEqual(report["complete_block"]["version"], 273)

    def test_accepts_frozen_reuse_finite_output_logprobs_alias(self):
        api = load_module()
        completed = final(273, "B")
        reuse = completed["results"][-1]
        reuse["finite_output_logprobs"] = reuse.pop("finite_logprobs")
        summary = api._summarize_process(completed, resources={})
        self.assertEqual(summary["cancel_reuse"], "PASS")

    def test_rejects_conflicting_logprob_finiteness_fields(self):
        api = load_module()
        completed = final(273, "B")
        reuse = completed["results"][-1]
        reuse["finite_output_logprobs"] = False
        with self.assertRaisesRegex(RuntimeError, "cancel/reuse contract"):
            api._summarize_process(completed, resources={})


if __name__ == "__main__":
    unittest.main()
