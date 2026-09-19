"""Launch intent regression; actual ServerArgs parsing is separately exercised."""
import ast
from pathlib import Path
import unittest


class LaunchScope(unittest.TestCase):
    def test_no_encoder_disaggregation(self):
        source=Path(__file__).resolve().parents[1]/'tools/prepare_q64_launch.py'
        values={n.value for n in ast.walk(ast.parse(source.read_text()))
                if isinstance(n,ast.Constant) and isinstance(n.value,str)}
        self.assertNotIn('--language-only',values)
        self.assertNotIn('--encoder-only',values)
        self.assertIn('--disable-cuda-graph',values)
        self.assertIn('--disable-shared-experts-fusion',values)
        self.assertIn('--kt-max-deferred-experts-per-token',values)
