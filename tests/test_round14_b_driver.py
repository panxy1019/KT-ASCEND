import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "artifacts/round14/round14_b_driver.py"
FIXTURE = ROOT / "artifacts/round14/ROUND14_B_REQUESTS.json"
EDGE = ROOT / "artifacts/round14/parents/round13-final-v228/round11-t1-fixture.json"


def load_module():
    spec = importlib.util.spec_from_file_location("round14_b_driver", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Round14BDriverTest(unittest.TestCase):
    def setUp(self):
        self.fixture = json.loads(FIXTURE.read_text())
        self.edge = json.loads(EDGE.read_text())

    def test_h512_schedule_is_bounded_balanced_and_exactly_64(self):
        api = load_module()
        for version, block in (
            (243, "A"), (244, "B"), (257, "A"), (258, "B"),
            (261, "A"), (262, "B"), (265, "A"), (266, "B"),
            (269, "A"), (270, "B"),
        ):
            rows = api.build_schedule(
                self.fixture, self.edge, version=version, profile="H512", block=block
            )
            self.assertEqual(len(rows), 64)
            counts = {kind: sum(row["kind"] == kind for row in rows) for kind in {
                "quality", "performance", "boundary_accept", "boundary_reject",
                "cancel", "reuse",
            }}
            self.assertEqual(counts, {
                "quality": 48, "performance": 12, "boundary_accept": 1,
                "boundary_reject": 1, "cancel": 1, "reuse": 1,
            })
            quality = [row for row in rows if row["kind"] == "quality"]
            self.assertTrue(any(row["split"] == "HOLDOUT" for row in quality))
            self.assertTrue(all(row["input_tokens"] <= 512 for row in quality))
            accepted = next(row for row in rows if row["kind"] == "boundary_accept")
            rejected = next(row for row in rows if row["kind"] == "boundary_reject")
            self.assertEqual((accepted["input_tokens"], accepted["decode_tokens"]), (512, 256))
            self.assertGreater(rejected["input_tokens"], 512)

    def test_h768_two_blocks_cover_all_64_sessions(self):
        api = load_module()
        schedules = [
            api.build_schedule(
                self.fixture, self.edge, version=version, profile="H768", block=block
            )
            for version, block in ((245, "A"), (246, "B"))
        ]
        self.assertTrue(all(len(rows) == 64 for rows in schedules))
        quality_ids = {
            row["case_id"] for rows in schedules for row in rows
            if row["kind"] == "quality"
        }
        self.assertEqual(quality_ids, {row["request_id"] for row in self.fixture["rows"]})
        for rows in schedules:
            accepted = next(row for row in rows if row["kind"] == "boundary_accept")
            rejected = next(row for row in rows if row["kind"] == "boundary_reject")
            self.assertEqual((accepted["input_tokens"], accepted["decode_tokens"]), (768, 256))
            self.assertGreater(rejected["input_tokens"], 768)
            self.assertEqual(sum(row["kind"] == "cancel" for row in rows), 1)
            self.assertEqual(sum(row["kind"] == "reuse" for row in rows), 1)
        repaired = [
            api.build_schedule(
                self.fixture, self.edge, version=version, profile="H768", block=block
            )
            for version, block in ((259, "A"), (260, "B"))
        ]
        self.assertTrue(all(len(rows) == 64 for rows in repaired))
        final = [
            api.build_schedule(
                self.fixture, self.edge, version=version, profile="H768", block=block
            )
            for version, block in ((263, "A"), (264, "B"))
        ]
        self.assertTrue(all(len(rows) == 64 for rows in final))
        evidence_complete = [
            api.build_schedule(
                self.fixture, self.edge, version=version, profile="H768", block=block
            )
            for version, block in ((267, "A"), (268, "B"))
        ]
        self.assertTrue(all(len(rows) == 64 for rows in evidence_complete))

    def test_qsa_validator_proves_prompt_chunks_decode_steps_and_drain(self):
        api = load_module()
        rid = "r"
        layers = list(range(3, 48, 4))
        events = []
        for layer in layers:
            for count in (32, 32, 32, 32):
                events.append({
                    "status": "CONTEXT_REFERENCE_QSA_DISPATCH",
                    "request_id": rid, "layer": layer,
                    "strategy": "VECTORIZED", "query_tokens": count,
                    "allocation_epoch": 7,
                })
            # The first output token is selected from the Prefill logits.  Only
            # the remaining completion tokens execute Decode forwards.
            for _ in range(15):
                events.append({
                    "status": "CONTEXT_REFERENCE_QSA_DISPATCH",
                    "request_id": rid, "layer": layer,
                    "strategy": "DIRECT_BF16_ROW", "query_tokens": 1,
                    "allocation_epoch": 7,
                })
        events.append({
            "status": "CONTEXT_REFERENCE_REQUEST_DRAINED",
            "request_id": rid, "allocation_epoch": 7, "hbm_release": False,
        })
        receipt = api.validate_qsa_state(rid, events, input_tokens=128, decode_tokens=16)
        self.assertEqual(receipt["prefill_tokens_per_layer"], 128)
        self.assertEqual(receipt["decode_steps_per_layer"], 15)
        self.assertEqual(receipt["allocation_epoch"], 7)
        broken = events[:-2] + [events[-1]]
        with self.assertRaisesRegex(RuntimeError, "Decode"):
            api.validate_qsa_state(rid, broken, input_tokens=128, decode_tokens=16)

    def test_qsa_validator_accounts_for_one_prompt_tail_direct_forward(self):
        api = load_module()
        rid = "prompt-tail"
        events = []
        for layer in range(3, 48, 4):
            for count in (32, 32, 32, 32):
                events.append({
                    "status": "CONTEXT_REFERENCE_QSA_DISPATCH",
                    "request_id": rid, "layer": layer,
                    "strategy": "VECTORIZED", "query_tokens": count,
                    "allocation_epoch": 9,
                })
            # One DIRECT row consumes the 129th prompt token; the other 15
            # are generated-token Decode forwards for a 16-token completion.
            for _ in range(16):
                events.append({
                    "status": "CONTEXT_REFERENCE_QSA_DISPATCH",
                    "request_id": rid, "layer": layer,
                    "strategy": "DIRECT_BF16_ROW", "query_tokens": 1,
                    "allocation_epoch": 9,
                })
        events.append({
            "status": "CONTEXT_REFERENCE_REQUEST_DRAINED",
            "request_id": rid, "allocation_epoch": 9, "hbm_release": False,
        })
        receipt = api.validate_qsa_state(
            rid, events, input_tokens=129, decode_tokens=16
        )
        self.assertEqual(receipt["prefill_tokens_per_layer"], 128)
        self.assertEqual(receipt["prompt_direct_steps_per_layer"], 1)
        self.assertEqual(receipt["decode_steps_per_layer"], 15)
        self.assertEqual(receipt["effective_context_forwards_per_layer"], 144)

    def test_known_answer_failure_is_recorded_without_stopping_distribution(self):
        api = load_module()
        row = {
            "case_id": "known-answer",
            "validator": {"kind": "exact_text", "expected": "ORBIT-01"},
        }
        value = {
            "text": "The answer is ORBIT-01",
            "output_ids": [1, 2],
            "meta_info": {
                "completion_tokens": 2,
                "output_token_logprobs": [[-0.1, 1], [-0.2, 2]],
                "finish_reason": {"type": "stop"},
            },
        }
        result = api.validate_quality(row, value)
        self.assertEqual(result["quality_status"], "FAIL_AUTOMATIC")
        self.assertIs(result["quality_pass"], False)
        self.assertEqual(result["output_text"], "The answer is ORBIT-01")


if __name__ == "__main__":
    unittest.main()
