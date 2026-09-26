import copy
import hashlib
import json
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from round11_t1_protocol import (  # noqa: E402
    CASE_LENGTHS,
    SCHEDULES,
    build_schedule,
    expected_chunks,
    sha256_json,
    validate_fixture,
    validate_cancel_receipt,
    validate_path,
    validate_response,
)


def fixture():
    cases = {}
    for name, length in CASE_LENGTHS.items():
        ids = list(range(length or 12))
        messages = [{"role": "user", "content": name}]
        rendered = "rendered-" + name
        validator = (
            {"kind": "abort"}
            if name == "cancel256"
            else {"kind": "exact_text", "expected": "391"}
            if name == "canary_math391"
            else {
                "kind": "assistant_message_prefix",
                "expected_text": "ACK",
                "expected_token_ids": [3897, 248046],
            }
        )
        if name.startswith("canary_") and name != "canary_math391":
            validator = {"kind": "exact_text", "expected": "OK"}
        cases[name] = {
            "input_tokens": len(ids),
            "input_ids": ids,
            "input_ids_sha256": sha256_json(ids),
            "retokenized_input_ids_sha256": sha256_json(ids),
            "messages": messages,
            "messages_sha256": sha256_json(messages),
            "rendered_chat": rendered,
            "rendered_chat_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
            "tokenization_method": "ACTUAL_TOKENIZER_APPLY_CHAT_TEMPLATE",
            "direct_token_edit": False,
            "validator": validator,
        }
    return {
        "schema": 1,
        "status": "FROZEN_ROUND11_T1_ACTUAL_TOKENIZER_FIXTURE",
        "model_revision": "de4b8e4d43b917e7706784d8bb445c9af86a3540",
        "chunk_tokens": 32,
        "tokenizer": {
            "chat_template_sha256": "a" * 64,
            "tokenizer_files_sha256": {
                "tokenizer.json": "b" * 64,
                "tokenizer_config.json": "c" * 64,
            },
            "add_generation_prompt": True,
            "enable_thinking": False,
            "add_special_tokens": False,
        },
        "cases": cases,
    }


def _path_rows(row):
    chunks = expected_chunks(len(row["request"]["input_ids"]))
    hybrid = [
        {
            "request_id": row["request"]["rid"],
            "request_tokens": len(row["request"]["input_ids"]),
            "batch_tokens": count,
            "start": start,
            "threshold": row["threshold"],
            "path": row["expected_path"],
            "basis": "REQUEST_TOTAL",
            "stage": "SELECTION",
        }
        for start, count in chunks
    ]
    if row["expected_path"] == "HYBRID":
        for start, count in chunks:
            for layer in (0, 3):
                for stage in (
                    "ORIGINAL_HYBRID_APPLY",
                    "CPU_SUBMIT",
                    "CPU_RESULT",
                    "NPU_RESIDENT_APPLY",
                ):
                    hybrid.append(
                        {
                            "request_id": row["request"]["rid"],
                            "start": start,
                            "batch_tokens": count,
                            "layer": layer,
                            "stage": stage,
                        }
                    )
        return hybrid, [], []
    layerwise = [
        {"request_id": row["request"]["rid"], "status": "LAYERWISE_BATCH_BEGIN", "start": start, "count": count}
        for start, count in chunks
    ] + [{"request_id": row["request"]["rid"], "status": "PREFILL_COMPLETE"}]
    streaming = []
    for start, count in chunks:
        streaming.extend(
            [
                {"request_id": row["request"]["rid"], "status": "STREAMING_CHUNK_BEGIN", "start": start, "count": count},
                {"request_id": row["request"]["rid"], "status": "STREAMING_CHUNK_DRAINED", "start": start, "count": count},
            ]
        )
    streaming.append(
        {
            "request_id": row["request"]["rid"],
            "status": "STREAMING_PREFILL_COMPLETE",
            "tokens": len(row["request"]["input_ids"]),
            "chunks": len(chunks),
        }
    )
    return hybrid, layerwise, streaming


class Round11T1ProtocolTests(unittest.TestCase):
    def test_schedules_freeze_boundary_and_hierarchical_search(self):
        document = fixture()
        validate_fixture(document)
        boundary = build_schedule(document, "BOUNDARY")
        self.assertEqual(
            [row["expected_path"] for row in boundary],
            ["HYBRID", "LAYERWISE", "LAYERWISE"],
        )
        for process in ("HYBRID256", "HYBRID512", "HYBRID1024"):
            self.assertTrue(
                all(
                    row["expected_path"] == "HYBRID"
                    for row in build_schedule(document, process)
                )
            )
        self.assertEqual(len(SCHEDULES["HYBRID256"]), 15)
        self.assertEqual(
            sum(row[1] == "cancel" for row in SCHEDULES["HYBRID256"]), 1
        )
        self.assertEqual(
            SCHEDULES["HYBRID_CANCEL_REPAIR"],
            (
                ("cancel256", "cancel", 8, False),
                ("canary_math391", "reuse", 32, False),
            ),
        )

    def test_path_contract_covers_real_selection_and_provider(self):
        for process, index in (
            ("BOUNDARY", 0),
            ("BOUNDARY", 1),
            ("HYBRID1024", 1),
        ):
            with self.subTest(process=process, index=index):
                row = build_schedule(fixture(), process)[index]
                evidence = _path_rows(row)
                result = validate_path(row, *evidence)
                self.assertEqual(result["path"], row["expected_path"])
                self.assertEqual(
                    sum(result["actual_dispatch_tokens"]),
                    len(row["request"]["input_ids"]),
                )

    def test_decode_exact_requires_full_length_and_finite_logprobs(self):
        row = next(
            row
            for row in build_schedule(fixture(), "HYBRID256")
            if row["phase"] == "decode_exact"
        )
        count = row["request"]["sampling_params"]["max_new_tokens"]
        value = {
            "text": "x" * count,
            "output_ids": list(range(count)),
            "meta_info": {
                "id": row["request"]["rid"],
                "prompt_tokens": len(row["request"]["input_ids"]),
                "cached_tokens": 0,
                "completion_tokens": count,
                "output_token_logprobs": [
                    [-0.1, index, None] for index in range(count)
                ],
                "finish_reason": {"type": "length", "length": count},
            },
        }
        self.assertEqual(validate_response(row, value)["completion_tokens"], count)
        broken = copy.deepcopy(value)
        broken["meta_info"]["output_token_logprobs"][0][0] = float("nan")
        with self.assertRaisesRegex(RuntimeError, "nonfinite"):
            validate_response(row, broken)

    def test_fixture_rejects_direct_token_edit_and_length_drift(self):
        document = fixture()
        document["cases"]["edge512"]["direct_token_edit"] = True
        with self.assertRaisesRegex(ValueError, "direct token edit"):
            validate_fixture(document)
        document = fixture()
        document["cases"]["edge512"]["input_ids"].pop()
        with self.assertRaisesRegex(ValueError, "exact token length"):
            validate_fixture(document)

    def test_cancel_receipt_uses_real_submit_fence_and_orders_lifecycle(self):
        receipt = {
            "status": "PASS_HYBRID_MIDPREFILL_CANCEL_AND_REUSE",
            "request_id": "cancel",
            "trigger_start": 32,
            "abort_http_status": 200,
            "generation_thread_alive": False,
            "terminal_finish_reason": "abort",
            "hybrid_events": [
                {"request_id": "cancel", "stage": "SELECTION", "start": 0},
                {"request_id": "cancel", "stage": "SELECTION", "start": 32},
            ],
            "cancel_events": [
                {"request_id": "cancel", "request_epoch": 13, "allocation_epoch": 13,
                 "event_seq": 86, "status": "CANCEL_SEEN"},
                {"request_id": "cancel", "request_epoch": 13, "allocation_epoch": 13,
                 "event_seq": 87, "status": "SUBMIT_FENCE_PUBLISHED"},
                {"request_id": "cancel", "request_epoch": 13, "allocation_epoch": 13,
                 "event_seq": 88, "status": "REQUEST_RETIRED"},
                {"request_id": "cancel", "request_epoch": 13, "allocation_epoch": 13,
                 "event_seq": 89, "status": "CANCEL_CLEANUP_COMMITTED"},
            ],
            "cancel_statuses": [
                "CANCEL_SEEN", "SUBMIT_FENCE_PUBLISHED",
                "REQUEST_RETIRED", "CANCEL_CLEANUP_COMMITTED",
            ],
            "late_commit": False,
            "lease_returned": True,
            "reuse": {"validator_passed": True, "request_id": "reuse"},
        }
        self.assertIs(validate_cancel_receipt(receipt), receipt)

        bad = copy.deepcopy(receipt)
        bad["cancel_events"][2]["event_seq"] = 85
        with self.assertRaisesRegex(RuntimeError, "order"):
            validate_cancel_receipt(bad)

        bad = copy.deepcopy(receipt)
        bad["hybrid_events"].append(
            {"request_id": "cancel", "stage": "SELECTION", "start": 64}
        )
        with self.assertRaisesRegex(RuntimeError, "post-cancel chunk"):
            validate_cancel_receipt(bad)


if __name__ == "__main__":
    unittest.main()
