import sys
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch


class ContextReferenceBindingTests(unittest.TestCase):
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
