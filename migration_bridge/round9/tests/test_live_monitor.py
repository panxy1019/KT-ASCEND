import ast
from pathlib import Path
import re
import unittest

source = Path(__file__).parents[1] / 'tools/monitor_owned_server.py'
tree = ast.parse(source.read_text())
nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'parse_physical15']
ns = {'re': re}
exec(compile(ast.Module(nodes, type_ignores=[]), str(source), 'exec'), ns)

class MonitorParserTests(unittest.TestCase):
    def test_physical15_only_and_correct_hbm_column(self):
        raw = '''| 0     14 | 0000:81:00.0 | 50 | 0 / 0 | 9391 / 65536 |
| 1     15 | 0000:83:00.0 | 25 | 0 / 0 | 8969 / 65536 |
| 7       0 | 4069719 | python3.10 | 6124 | NA |
| 7       1 | 4069720 | python3.10 | 6124 | NA |'''
        self.assertEqual(ns['parse_physical15'](raw), (8969, {4069720: 6124}))

    def test_missing_or_wrong_capacity_rejected(self):
        for raw in ('', '| 1 15 | 0000:83:00.0 | 0 / 0 | 100 / 32768 |'):
            with self.assertRaises(RuntimeError):
                ns['parse_physical15'](raw)

if __name__ == '__main__':
    unittest.main()
