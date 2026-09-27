import sys
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch


class ContextReferenceBindingTests(unittest.TestCase):
    def test_qsa_binding_arguments_are_absent_for_the_qualified_source(self):
        from migration_bridge.round9.runtime.context_reference_binding import (
            _qsa_binding_arguments,
        )
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(_qsa_binding_arguments(Path("unused")), {})

    def test_qsa_binding_arguments_require_a_complete_proof_pair_and_hash_consumer(self):
        from migration_bridge.round9.runtime.context_reference_binding import (
            _qsa_binding_arguments,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proof = root / "proof.json"
            proof.write_text("{}")
            consumer = root / "qwen4_exp.py"
            consumer.write_text("candidate consumer")
            runtime = root / "context_reference_runtime.py"
            runtime.write_text("candidate runtime")
            binding = root / "context_reference_binding.py"
            binding.write_text("candidate binding")
            digest = hashlib.sha256(proof.read_bytes()).hexdigest()
            with patch.dict("os.environ", {
                "ROUND9_QSA_SOURCE_BINDING": str(proof),
                "ROUND9_QSA_SOURCE_BINDING_SHA256": digest,
            }, clear=True):
                self.assertEqual(_qsa_binding_arguments(
                    consumer, runtime_path=runtime, binding_path=binding), {
                    "qsa_source_binding_path": str(proof),
                    "qsa_source_binding_sha256": digest,
                    "qsa_consumer_sha256": hashlib.sha256(
                        consumer.read_bytes()).hexdigest(),
                    "qsa_runtime_source_sha256": hashlib.sha256(
                        runtime.read_bytes()).hexdigest(),
                    "qsa_binding_source_sha256": hashlib.sha256(
                        binding.read_bytes()).hexdigest(),
                })
            with patch.dict("os.environ", {
                "ROUND9_QSA_SOURCE_BINDING": str(proof),
            }, clear=True):
                with self.assertRaisesRegex(RuntimeError, "path/hash pair"):
                    _qsa_binding_arguments(consumer)

    def test_disabled_prepare_ready_and_hooks_do_not_touch_objects(self):
        from migration_bridge.round9.runtime.context_reference_binding import (
            context_reference_prepare, context_reference_ready,
            context_before, context_after,
        )
        with patch(
                "migration_bridge.round9.runtime.context_reference_binding.load_context_reference_profile",
                return_value=None):
            self.assertIsNone(context_reference_prepare(object()))
            self.assertIsNone(context_reference_ready(object()))
            self.assertIsNone(context_before(object(), object()))
            self.assertIsNone(context_after(object()))

    def test_profile_guard_rejects_nonserial_scheduler_before_device_import(self):
        from migration_bridge.round9.runtime.context_reference_binding import context_reference_prepare
        profile = {"mode": "BF16_COMPACT", "device_gate": {}}
        args = NS(disable_overlap_schedule=False)
        with patch(
                "migration_bridge.round9.runtime.context_reference_binding.load_context_reference_profile",
                return_value=profile):
            with self.assertRaisesRegex(RuntimeError, "serial"):
                context_reference_prepare(NS(server_args=args))

    def test_after_allows_scheduler_idle_but_rejects_unleased_active_runtime(self):
        from migration_bridge.round9.runtime.context_reference_binding import context_after
        runtime = NS(active=False, request_token=None)
        runner = NS(_round9_context_reference=runtime, _round9_context_request=None)
        scheduler = NS(tp_worker=NS(model_runner=runner))
        with patch(
                "migration_bridge.round9.runtime.context_reference_binding.load_context_reference_profile",
                return_value={"mode": "BF16_COMPACT"}):
            self.assertIsNone(context_after(scheduler))
            runtime.active = True
            with self.assertRaisesRegex(RuntimeError, "unleased"):
                context_after(scheduler)

    def test_overlay_patch_retains_sync_reference_and_installs_optional_consumer(self):
        tools = Path(__file__).resolve().parents[1] / "tools"
        sys.path.insert(0, str(tools))
        try:
            from install_framework_overlay import patch_context_consumer
            source = (Path(__file__).resolve().parents[1]
                      / "sglang_overlay/python/sglang/srt/models/qwen4_exp.py").read_text()
            modified = patch_context_consumer(source)
            compile(modified, "qwen4_exp.py", "exec")
            self.assertIn("context_consumer.consume(", modified)
            self.assertIn("else sparse_gqa(q, keys, values, selected)", modified)
            with self.assertRaises(ValueError):
                patch_context_consumer(modified)
        finally:
            sys.path.remove(str(tools))


if __name__ == "__main__":
    unittest.main()
