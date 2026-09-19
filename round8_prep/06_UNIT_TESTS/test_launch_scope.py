"""Launch intent regression; actual ServerArgs parsing is separately exercised."""
import ast
from pathlib import Path
import unittest
import sys
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from qwen_text_server import configure_text


class LaunchScope(unittest.TestCase):
    def test_programmatic_text_switch(self):
        args=SimpleNamespace(language_only=False,encoder_only=False,enable_multimodal=None,model_path='frozen')
        self.assertIs(configure_text(args),args)
        self.assertFalse(args.enable_multimodal)
        self.assertEqual(args.model_path,'frozen')
        args.language_only=True
        with self.assertRaises(AssertionError):configure_text(args)
    def test_no_encoder_disaggregation(self):
        source=Path(__file__).resolve().parents[1]/'tools/prepare_q64_launch.py'
        values={n.value for n in ast.walk(ast.parse(source.read_text()))
                if isinstance(n,ast.Constant) and isinstance(n.value,str)}
        self.assertNotIn('--language-only',values)
        self.assertNotIn('--encoder-only',values)
        self.assertIn('--disable-cuda-graph',values)
        self.assertIn('--disable-shared-experts-fusion',values)
        self.assertIn('--kt-max-deferred-experts-per-token',values)
