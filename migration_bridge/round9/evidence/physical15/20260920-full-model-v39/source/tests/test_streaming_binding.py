import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch


class StreamingBindingTests(unittest.TestCase):
    def test_stack_contract_requires_all_planes_and_matching_threshold(self):
        from migration_bridge.round9.runtime.streaming_binding import validate_streaming_stack
        streaming = {"schema": 1, "threshold_tokens": 4, "chunk_tokens": 32}
        framework = {"prefill": {"threshold_tokens": 4}, "ple": {}}
        self.assertEqual(
            validate_streaming_stack(streaming, framework, {"mode": "BF16_COMPACT"}),
            streaming,
        )
        for bad_framework, context in (
                (None, {"mode": "BF16_COMPACT"}),
                ({"prefill": None, "ple": {}}, {"mode": "BF16_COMPACT"}),
                ({"prefill": {"threshold_tokens": 5}, "ple": {}}, {"mode": "BF16_COMPACT"}),
                (framework, None)):
            with self.assertRaisesRegex(ValueError, "three-plane streaming"):
                validate_streaming_stack(streaming, bad_framework, context)

    def test_checked_in_streaming_profiles_form_one_valid_stack(self):
        from migration_bridge.round9.runtime.framework_config import parse_framework
        from migration_bridge.round9.runtime.context_reference_runtime import (
            parse_context_reference_profile,
        )
        from migration_bridge.round9.runtime.streaming_binding import (
            parse_streaming_profile, validate_streaming_stack,
        )
        profiles = Path(__file__).resolve().parents[1] / "runtime/profiles"
        streaming = parse_streaming_profile(json.loads(
            (profiles / "streaming_bounded32_v1.json").read_text()))
        framework = parse_framework(json.loads(
            (profiles / "framework_streaming_q32_v1.json").read_text()))
        context = parse_context_reference_profile(json.loads(
            (profiles / "context_reference_bf16_compact_v1.json").read_text()))
        self.assertEqual(validate_streaming_stack(streaming, framework, context), streaming)

    def test_profile_hash_and_strict_fields(self):
        from migration_bridge.round9.runtime.streaming_binding import (
            parse_streaming_profile, load_streaming_profile)
        profile = {"schema": 1, "threshold_tokens": 4, "chunk_tokens": 32}
        self.assertEqual(parse_streaming_profile(profile), profile)
        for override in ({"schema": True}, {"threshold_tokens": 0},
                         {"chunk_tokens": 16}, {"extra": True}):
            value = dict(profile); value.update(override)
            with self.assertRaises(ValueError): parse_streaming_profile(value)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"; path.write_text(json.dumps(profile))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            env = {"ROUND9_STREAMING_PROFILE": str(path),
                   "ROUND9_STREAMING_SHA256": digest}
            with patch.dict("os.environ", env, clear=True):
                self.assertEqual(load_streaming_profile(), profile)
                path.write_text("{}")
                with self.assertRaisesRegex(ValueError, "changed"):
                    load_streaming_profile()

    def test_default_off_and_missing_plane_binding_fail_closed(self):
        from migration_bridge.round9.runtime.streaming_binding import streaming_model_ready
        with patch("migration_bridge.round9.runtime.streaming_binding.load_streaming_profile",
                   return_value=None):
            self.assertIsNone(streaming_model_ready(object()))
        profile = {"schema": 1, "threshold_tokens": 4, "chunk_tokens": 32}
        framework = {"prefill": {"threshold_tokens": 4}, "ple": {}}
        runner = NS(_round9_prefill=object(), _round9_ple=object(),
                    _round9_context_reference_profile={"mode": "BF16_COMPACT"})
        with patch("migration_bridge.round9.runtime.streaming_binding.load_streaming_profile",
                   return_value=profile), patch(
                   "migration_bridge.round9.runtime.streaming_binding.load_framework",
                   return_value=framework):
            with self.assertRaisesRegex(RuntimeError, "three bound planes"):
                streaming_model_ready(runner)


if __name__ == "__main__":
    unittest.main()
