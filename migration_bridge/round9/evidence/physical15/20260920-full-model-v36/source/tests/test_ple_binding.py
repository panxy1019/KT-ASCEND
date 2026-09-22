import sys
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch


class PLEBindingTests(unittest.TestCase):
    def test_disabled_binding_never_touches_model_or_imports_device(self):
        from migration_bridge.round9.runtime.ple_binding import ple_model_ready
        with patch('migration_bridge.round9.runtime.ple_binding.load_framework',return_value=None):
            self.assertIsNone(ple_model_ready(object()))

    def test_overlap_rejected_before_device_import(self):
        from migration_bridge.round9.runtime.ple_binding import ple_model_ready
        with patch('migration_bridge.round9.runtime.ple_binding.load_framework',return_value={'ple':{'mode':'ASYNC'}}):
            with self.assertRaisesRegex(RuntimeError,'serial'):
                ple_model_ready(NS(server_args=NS(disable_overlap_schedule=False)))

    def test_prepare_seam_retains_synchronous_reference_and_commits_history_late(self):
        tools=Path(__file__).parents[1]/'tools';sys.path.insert(0,str(tools))
        try:
            from install_framework_overlay import patch_ple_prepare
            source=(Path(__file__).parents[1]/'sglang_overlay/python/sglang/srt/models/qwen4_exp.py').read_text()
            modified=patch_ple_prepare(source);compile(modified,'qwen4_exp.py','exec')
            self.assertIn('else self._table.gather_rows(row_ids)',modified)
            self.assertEqual(modified.count('req_pool.set_ngram_context(ple_data[0], ple_data[2])'),1)
            self.assertGreater(modified.index('req_pool.set_ngram_context(ple_data[0], ple_data[2])'),modified.index('for i, layer in enumerate(self.layers):'))
            with self.assertRaises(ValueError):patch_ple_prepare(modified)
        finally:sys.path.remove(str(tools))


if __name__ == '__main__':unittest.main()
