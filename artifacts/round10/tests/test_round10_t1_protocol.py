import copy
import hashlib
import json
import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import round10_t1_protocol as protocol


def digest(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def case(name, length, validator):
    ids = list(range(1000, 1000 + length))
    messages = [{"role": "user", "content": f"fixture {name}"}]
    rendered = f"rendered {name}"
    return {
        "input_tokens": length,
        "input_ids": ids,
        "input_ids_sha256": digest(ids),
        "messages": messages,
        "messages_sha256": digest(messages),
        "rendered_chat": rendered,
        "rendered_chat_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
        "retokenized_input_ids_sha256": digest(ids),
        "tokenization_method": "ACTUAL_TOKENIZER_APPLY_CHAT_TEMPLATE",
        "direct_token_edit": False,
        "validator": validator,
    }


def fixture():
    exact = lambda text: {"kind": "exact_text", "expected": text}
    ack = {"kind": "assistant_message_prefix", "expected_text": "ACK", "expected_token_ids": [3897, 248046]}
    rows = {
        "canary_marker": case("canary_marker", 25, exact("R10-HYBRID-OK")),
        "canary_math391": case("canary_math391", 27, exact("391")),
        "canary_json": case("canary_json", 28, exact('{"ok":true}')),
        "canary_quote": case("canary_quote", 26, exact("cobalt")),
        "canary_transform": case("canary_transform", 30, exact("ABCXYZ")),
        "edge63": case("edge63", 63, ack),
        "edge64": case("edge64", 64, ack),
        "edge65": case("edge65", 65, ack),
        "edge127": case("edge127", 127, ack),
        "edge128": case("edge128", 128, ack),
        "policy129": case("policy129", 129, ack),
        "policy130": case("policy130", 130, ack),
        "decode64": case("decode64", 64, {"kind": "multi_step", "minimum_tokens": 8}),
        "cancel128": case("cancel128", 128, {"kind": "abort"}),
    }
    return {
        "schema": 1,
        "status": "FROZEN_ROUND10_T1_ACTUAL_TOKENIZER_FIXTURE",
        "model_revision": "d" * 40,
        "threshold_tokens": 129,
        "chunk_tokens": 32,
        "tokenizer": {
            "tokenizer_files_sha256": {"tokenizer.json": "a" * 64, "tokenizer_config.json": "b" * 64},
            "chat_template_sha256": "c" * 64,
            "add_generation_prompt": True,
            "enable_thinking": False,
            "add_special_tokens": False,
        },
        "cases": rows,
    }


class FixtureTests(unittest.TestCase):
    def test_fixture_contract_accepts_exact_case_set_and_lengths(self):
        got = protocol.validate_fixture(fixture())
        self.assertEqual(got["cases"]["edge128"]["input_tokens"], 128)

    def test_fixture_rejects_direct_token_edit_or_hash_drift(self):
        bad = fixture()
        bad["cases"]["edge127"]["direct_token_edit"] = True
        with self.assertRaisesRegex(ValueError, "direct token edit"):
            protocol.validate_fixture(bad)
        bad = fixture()
        bad["cases"]["canary_json"]["input_ids_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "receipt hash"):
            protocol.validate_fixture(bad)


class ScheduleTests(unittest.TestCase):
    def test_process_a_has_exact_eleven_generation_slots(self):
        rows = protocol.build_schedule(fixture(), "A")
        self.assertEqual(len(rows), 11)
        self.assertEqual([row["phase"] for row in rows], [
            "canary", "canary", "canary", "canary", "canary",
            "warmup", "measurement", "measurement", "decode", "cancel", "reuse",
        ])
        self.assertEqual(sum(row["phase"] == "cancel" for row in rows), 1)
        self.assertTrue(all(row["expected_path"] == "HYBRID" for row in rows))

    def test_process_b_has_exact_eleven_generation_slots_and_all_edges(self):
        rows = protocol.build_schedule(fixture(), "B")
        self.assertEqual(len(rows), 11)
        self.assertEqual([row["case_id"] for row in rows[:5]], [
            "edge63", "edge64", "edge65", "edge127", "edge128",
        ])
        self.assertEqual([row["phase"] for row in rows[-3:]], ["measurement", "cancel", "reuse"])
        self.assertTrue(all(row["expected_path"] == "HYBRID" for row in rows))

    def test_recovery_process_fits_remaining_budget_and_covers_both_cancels(self):
        rows = protocol.build_schedule(fixture(), "R")
        self.assertEqual(len(rows), 19)
        self.assertEqual(sum(row["phase"] == "cancel" for row in rows), 2)
        self.assertEqual(sum(row["phase"] == "reuse" for row in rows), 2)
        self.assertEqual(
            {row["case_id"] for row in rows if row["phase"] == "edge"},
            {"edge63", "edge64", "edge65", "edge127", "edge128"},
        )
        self.assertEqual(
            [row["request"]["sampling_params"]["max_new_tokens"] for row in rows if row["phase"] == "decode"],
            [32, 64],
        )

    def test_policy_is_request_total_extend_only_with_inclusive_layerwise_edge(self):
        self.assertEqual(protocol.select_path(total=128, forward_mode="EXTEND"), "HYBRID")
        self.assertEqual(protocol.select_path(total=129, forward_mode="EXTEND"), "LAYERWISE")
        self.assertEqual(protocol.select_path(total=130, forward_mode="EXTEND"), "LAYERWISE")
        self.assertEqual(protocol.select_path(total=130, forward_mode="DECODE"), "HYBRID")
        with self.assertRaises(ValueError):
            protocol.select_path(total=0, forward_mode="EXTEND")


class ResponseTests(unittest.TestCase):
    def response(self, row, text, ids, finish="stop"):
        return {
            "text": text,
            "output_ids": ids,
            "meta_info": {
                "id": row["request"]["rid"],
                "prompt_tokens": len(row["request"]["input_ids"]),
                "cached_tokens": 0,
                "completion_tokens": len(ids),
                "output_token_logprobs": [[-0.1, token, None] for token in ids],
                "finish_reason": {"type": finish},
            },
        }

    def test_known_answer_and_multistep_contracts(self):
        canary = protocol.build_schedule(fixture(), "A")[1]
        got = protocol.validate_response(canary, self.response(canary, "391", [18, 24, 16, 248046]))
        self.assertTrue(got["finite_output_logprobs"])
        decode = protocol.build_schedule(fixture(), "A")[8]
        got = protocol.validate_response(decode, self.response(decode, "many steps", list(range(8)), "stop"))
        self.assertEqual(got["completion_tokens"], 8)

    def test_nonfinite_or_wrong_known_answer_fails(self):
        row = protocol.build_schedule(fixture(), "A")[0]
        bad = self.response(row, "wrong", [1])
        with self.assertRaisesRegex(RuntimeError, "known-answer"):
            protocol.validate_response(row, bad)
        bad = self.response(row, "R10-HYBRID-OK", [1])
        bad["meta_info"]["output_token_logprobs"][0][0] = math.nan
        with self.assertRaisesRegex(RuntimeError, "nonfinite"):
            protocol.validate_response(row, bad)


class PathAndCancelTests(unittest.TestCase):
    def test_hybrid_path_requires_every_chunk_selection_and_real_providers(self):
        row = protocol.build_schedule(fixture(), "B")[4]
        events = []
        for start, count in ((0, 32), (32, 32), (64, 32), (96, 32)):
            base = {
                "request_id": row["request"]["rid"], "request_tokens": 128,
                "batch_tokens": count, "start": start, "threshold": 129,
                "path": "HYBRID", "basis": "REQUEST_TOTAL",
            }
            events.append(dict(base, stage="SELECTION"))
            for layer in (0, 3):
                for stage in protocol.HYBRID_PROVIDER_STAGES:
                    events.append(dict(base, layer=layer, stage=stage, outcome="RETURN"))
        got = protocol.validate_hybrid_path(row, events, [], [])
        self.assertEqual(got["chunks"], [[0, 32], [32, 32], [64, 32], [96, 32]])
        events.pop()
        with self.assertRaisesRegex(RuntimeError, "provider"):
            protocol.validate_hybrid_path(row, events, [], [])

    def test_cancel_receipt_requires_trigger_abort_no_later_chunk_and_cleanup(self):
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
            "cancel_statuses": ["CANCEL_SEEN", "CANCEL_CLEANUP_COMMITTED"],
            "late_commit": False,
            "lease_returned": True,
            "reuse": {"validator_passed": True, "request_id": "reuse"},
        }
        self.assertTrue(protocol.validate_cancel_receipt(receipt))
        bad = copy.deepcopy(receipt)
        bad["hybrid_events"].append({"request_id": "cancel", "stage": "SELECTION", "start": 64})
        with self.assertRaisesRegex(RuntimeError, "post-cancel chunk"):
            protocol.validate_cancel_receipt(bad)


if __name__ == "__main__":
    unittest.main()
