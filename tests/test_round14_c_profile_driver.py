import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "artifacts/round14/round14_c_profile_driver.py"
FIXTURE = ROOT / "artifacts/round14/parents/round13-final-v228/round11-t1-fixture.json"


def load_module():
    spec = importlib.util.spec_from_file_location("round14_c_profile_driver", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Round14CProfileDriverTest(unittest.TestCase):
    def test_schedule_is_exact_four_request_profile_contract(self):
        api = load_module()
        fixture = json.loads(FIXTURE.read_text())
        for version, block in ((249, "A"), (250, "B")):
            rows = api.build_schedule(fixture, version=version, block=block)
            self.assertEqual(len(rows), 4)
            self.assertEqual([row["phase"] for row in rows], [
                "warmup", "measurement", "measurement", "drain",
            ])
            self.assertEqual([row["workload"] for row in rows], [
                "WARMUP", "W2", "W3", "DRAIN",
            ])
            self.assertEqual(
                [(row["input_tokens"], row["decode_tokens"]) for row in rows],
                [(128, 16), (128, 128), (512, 16), (27, 4)],
            )

    def test_qsa_validation_requires_vectorized_prefill_and_direct_row_decode(self):
        api = load_module()
        rid = "profile"
        events = []
        for layer in api.QSA_LAYERS:
            events.append({
                "status": "CONTEXT_REFERENCE_QSA_DISPATCH", "request_id": rid,
                "layer": layer, "strategy": "VECTORIZED", "query_tokens": 128,
            })
            events.extend({
                "status": "CONTEXT_REFERENCE_QSA_DISPATCH", "request_id": rid,
                "layer": layer, "strategy": "DIRECT_BF16_ROW", "query_tokens": 1,
            } for _ in range(15))
        receipt = api.validate_qsa(rid, events, input_tokens=128, decode_tokens=16)
        self.assertEqual(receipt["prefill_strategy"], "VECTORIZED")
        self.assertEqual(receipt["decode_strategy"], "DIRECT_BF16_ROW")
        self.assertEqual(receipt["generated_decode_steps_per_layer"], 15)
        self.assertTrue(receipt["first_completion_token_from_prefill_logits"])
        broken = [dict(row) for row in events]
        broken[-1]["strategy"] = "ROW"
        with self.assertRaisesRegex(RuntimeError, "QSA"):
            api.validate_qsa(rid, broken, input_tokens=128, decode_tokens=16)

    def test_qsa_validation_accounts_for_one_token_prompt_tail(self):
        api = load_module()
        rid = "profile-tail"
        events = []
        for layer in api.QSA_LAYERS:
            events.extend((
                {
                    "status": "CONTEXT_REFERENCE_QSA_DISPATCH", "request_id": rid,
                    "layer": layer, "strategy": "VECTORIZED", "query_tokens": 32,
                },
                {
                    "status": "CONTEXT_REFERENCE_QSA_DISPATCH", "request_id": rid,
                    "layer": layer, "strategy": "DIRECT_BF16_ROW", "query_tokens": 1,
                },
            ))
            events.extend({
                "status": "CONTEXT_REFERENCE_QSA_DISPATCH", "request_id": rid,
                "layer": layer, "strategy": "DIRECT_BF16_ROW", "query_tokens": 1,
            } for _ in range(3))
        receipt = api.validate_qsa(rid, events, input_tokens=33, decode_tokens=4)
        self.assertEqual(receipt["prompt_direct_steps_per_layer"], 1)
        self.assertEqual(receipt["generated_decode_steps_per_layer"], 3)


if __name__ == "__main__":
    unittest.main()
