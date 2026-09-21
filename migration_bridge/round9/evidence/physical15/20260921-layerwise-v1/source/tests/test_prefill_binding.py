import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from migration_bridge.round9.runtime.prefill_binding import prefill_model_ready


class PrefillBindingTests(unittest.TestCase):
    def test_default_off_does_not_touch_model_or_import_device(self):
        with patch('migration_bridge.round9.runtime.prefill_binding.load_framework', return_value=None):
            self.assertIsNone(prefill_model_ready(object()))

    def test_model_runner_seam_precedes_kv_pool_and_is_unique(self):
        tools = Path(__file__).resolve().parents[1] / 'tools'
        sys.path.insert(0, str(tools))
        try:
            from install_framework_overlay import patch_model_runner
            original = 'class Runner:\n    def initialize(self):\n        self.load_model()\n        self.init_memory_pool()\n'
            result = patch_model_runner(original)
            compile(result, 'runner.py', 'exec')
            self.assertLess(result.index('prefill_model_ready(self)'), result.index('self.init_memory_pool()'))
            with self.assertRaises(ValueError): patch_model_runner(result)
            with self.assertRaises(ValueError): patch_model_runner('no valid seam')
        finally: sys.path.remove(str(tools))

    def test_unsupported_scheduler_rejected_before_device_import(self):
        config = dict(prefill=dict(threshold_tokens=1))
        runner = NS(server_args=NS(disable_overlap_schedule=False))
        with patch('migration_bridge.round9.runtime.prefill_binding.load_framework', return_value=config):
            with self.assertRaisesRegex(RuntimeError, 'serial'):
                prefill_model_ready(runner)


if __name__ == '__main__': unittest.main()
