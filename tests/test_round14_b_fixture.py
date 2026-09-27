import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "artifacts/round14/build_round14_b_fixture.py"


def load_module():
    spec = importlib.util.spec_from_file_location("round14_b_fixture", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_fixture(api):
    rows = []
    lengths = {"LE128": 96, "129_256": 192, "257_512": 384, "513_768": 640}
    for spec in api.build_specs():
        length = lengths[spec["bucket"]]
        rows.append({
            **spec,
            "input_tokens": length,
            "input_ids": list(range(length)),
            "rendered_chat": "rendered",
            "input_ids_sha256": api.sha256_json(list(range(length))),
            "messages_sha256": api.sha256_json(spec["messages"]),
            "rendered_chat_sha256": api.sha256_bytes(b"rendered"),
        })
    return {"schema": 1, "status": "FROZEN_ROUND14_B_PROXY_FIXTURE", "rows": rows}


class Round14BFixtureTest(unittest.TestCase):
    def test_specs_are_64_session_disjoint_balanced_and_traceable(self):
        api = load_module()
        rows = api.build_specs()
        self.assertEqual(len(rows), 64)
        self.assertEqual(sum(row["split"] == "DEV" for row in rows), 16)
        self.assertEqual(sum(row["split"] == "HOLDOUT" for row in rows), 48)
        self.assertEqual(len({row["session_id"] for row in rows}), 64)
        self.assertEqual(
            {bucket: sum(row["bucket"] == bucket for row in rows)
             for bucket in api.BUCKETS},
            {bucket: 16 for bucket in api.BUCKETS},
        )
        self.assertEqual(set(row["category"] for row in rows), set(api.CATEGORIES))
        self.assertTrue(all(row["provenance"] == api.PROVENANCE for row in rows))
        self.assertTrue(any(len(row["messages"]) > 1 for row in rows))

    def test_validator_accepts_frozen_lengths_and_rejects_leak_or_wrong_bucket(self):
        api = load_module()
        fixture = synthetic_fixture(api)
        self.assertTrue(api.validate_fixture(fixture))
        leaked = json.loads(json.dumps(fixture))
        leaked["rows"][16]["session_id"] = leaked["rows"][0]["session_id"]
        with self.assertRaisesRegex(ValueError, "session"):
            api.validate_fixture(leaked)
        wrong = json.loads(json.dumps(fixture))
        wrong["rows"][0]["input_tokens"] = 200
        wrong["rows"][0]["input_ids"] = list(range(200))
        wrong["rows"][0]["input_ids_sha256"] = api.sha256_json(list(range(200)))
        with self.assertRaisesRegex(ValueError, "bucket"):
            api.validate_fixture(wrong)


if __name__ == "__main__":
    unittest.main()
