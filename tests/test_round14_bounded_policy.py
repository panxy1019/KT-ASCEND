import importlib.util
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "artifacts/round14/round14_bounded_policy.py"


def load_module():
    spec = importlib.util.spec_from_file_location("round14_bounded_policy", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Round14BoundedPolicyTest(unittest.TestCase):
    def test_profile_couples_threshold_and_rejects_invalid_limits(self):
        api = load_module()
        valid = {
            "schema": 1, "mode": "H_BOUNDED", "max_input_tokens": 512,
            "max_output_tokens": 256, "max_total_tokens": 768,
            "hybrid_threshold_tokens": 513, "outside_policy": "REJECT",
        }
        self.assertEqual(api.parse_profile(valid)["max_input_tokens"], 512)
        for key, value in (("hybrid_threshold_tokens", 512), ("outside_policy", "LEGACY")):
            broken = dict(valid); broken[key] = value
            with self.assertRaisesRegex(ValueError, "profile"):
                api.parse_profile(broken)

    def test_accepts_inside_and_rejects_before_dispatch_with_receipt(self):
        api = load_module()
        profile = {
            "schema": 1, "mode": "H_BOUNDED", "max_input_tokens": 512,
            "max_output_tokens": 256, "max_total_tokens": 768,
            "hybrid_threshold_tokens": 513, "outside_policy": "REJECT",
        }
        with tempfile.TemporaryDirectory() as directory:
            event = Path(directory) / "events.jsonl"
            obj = types.SimpleNamespace(
                rid="inside", sampling_params={"max_new_tokens": 256}
            )
            self.assertTrue(api.enforce(profile, obj, list(range(512)), event_path=event))
            too_long = types.SimpleNamespace(
                rid="outside-input", sampling_params={"max_new_tokens": 1}
            )
            with self.assertRaisesRegex(ValueError, "H-BOUNDED rejected"):
                api.enforce(profile, too_long, list(range(513)), event_path=event)
            too_many_output = types.SimpleNamespace(
                rid="outside-output", sampling_params={"max_new_tokens": 257}
            )
            with self.assertRaisesRegex(ValueError, "H-BOUNDED rejected"):
                api.enforce(profile, too_many_output, list(range(128)), event_path=event)
            rows = [json.loads(line) for line in event.read_text().splitlines()]
            self.assertEqual([row["status"] for row in rows], [
                "ROUND14_H_BOUNDED_ACCEPT", "ROUND14_H_BOUNDED_REJECT",
                "ROUND14_H_BOUNDED_REJECT",
            ])
            self.assertTrue(all(row["before_scheduler_dispatch"] is True for row in rows))

    def test_environment_binding_requires_hash_and_is_default_off(self):
        api = load_module()
        obj = types.SimpleNamespace(rid="x", sampling_params={"max_new_tokens": 1})
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(api.enforce_from_environment(obj, [1]))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = root / "profile.json"
            profile.write_text(json.dumps({
                "schema": 1, "mode": "H_BOUNDED", "max_input_tokens": 512,
                "max_output_tokens": 256, "max_total_tokens": 768,
                "hybrid_threshold_tokens": 513, "outside_policy": "REJECT",
            }))
            with mock.patch.dict(os.environ, {
                "ROUND14_H_BOUNDED_PROFILE": str(profile),
                "ROUND14_H_BOUNDED_PROFILE_SHA256": "0" * 64,
                "ROUND14_H_BOUNDED_EVENTS": str(root / "events.jsonl"),
            }, clear=True):
                api.reset_for_test()
                with self.assertRaisesRegex(RuntimeError, "hash"):
                    api.enforce_from_environment(obj, [1])


if __name__ == "__main__":
    unittest.main()
