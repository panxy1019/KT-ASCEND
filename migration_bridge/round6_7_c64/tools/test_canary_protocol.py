"""Exercise startup assertions without importing accelerator dependencies."""
import ast
from pathlib import Path
import unittest


class CanaryProtocolTest(unittest.TestCase):
    def validate(self, manifest):
        tree = ast.parse(Path(__file__).with_name('run_canary.py').read_text())
        checks = [node for node in tree.body if isinstance(node, ast.Assert)
                  and 'manifest' in ast.unparse(node)]
        self.assertEqual(len(checks), 3)
        for node in checks:
            exec(compile(ast.Module(body=[node], type_ignores=[]), 'protocol', 'exec'),
                 {'manifest': manifest})

    def fixture(self):
        return dict(prompts=[dict(id=str(i)) for i in range(4)],
                    budgets=[16, 8, 8, 8, 8, 8, 4, 2],
                    restart_budget=16, observer_budget=8)

    def test_frozen_protocol(self):
        self.validate(self.fixture())

    def test_reject_protocol_drift(self):
        for key, value in [('budgets', [16, 8, 4, 2]), ('restart_budget', 8),
                           ('observer_budget', 16), ('prompts', [dict(id='x')] * 4)]:
            with self.subTest(key=key):
                manifest = self.fixture()
                manifest[key] = value
                with self.assertRaises(AssertionError):
                    self.validate(manifest)


if __name__ == '__main__':
    unittest.main()
