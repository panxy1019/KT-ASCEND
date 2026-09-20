import ast
from pathlib import Path
import re
import unittest

source = Path(__file__).parents[1] / 'tools/monitor_owned_server.py'
tree = ast.parse(source.read_text())
nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('parse_physical15','hbm_budget_exceeded','stop_owned','stable_system_baseline')]
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

class MonitorBudgetTests(unittest.TestCase):
    def test_async_board_process_counters_do_not_invent_training_growth(self):
        #Actual v4 sample during owned allocations; training stayed at6124MiB.
        self.assertFalse(ns['hbm_budget_exceeded'](28592,10864,6124,2845))

    def test_absolute_board_owned_and_training_bounds_remain_immediate(self):
        for values in [(56833,40000,6124,2845),(48000,41473,6124,2845),
                       (30000,10000,12516,2845)]:
            self.assertTrue(ns['hbm_budget_exceeded'](*values))

class MonitorFailureTests(unittest.TestCase):
    def test_output_failure_still_stops_only_owned_container(self):
        from types import SimpleNamespace as NS
        calls=[]
        class BadPath:
            def open(self,*args): raise OSError('simulated full log disk')
        ns.update(a=NS(output=BadPath(),container='kt-round9-full-model-v99'),
                  subprocess=NS(run=lambda cmd,**kw:calls.append(cmd)),
                  time=NS(time=lambda:1000),json=__import__('json'))
        with self.assertRaises(OSError): ns['stop_owned']('test failure')
        self.assertEqual(calls,[['docker','stop','--time','10','kt-round9-full-model-v99']])

    def test_baseline_needs_three_stable_created_samples(self):
        self.assertIsNone(ns['stable_system_baseline']([2845]))
        self.assertIsNone(ns['stable_system_baseline']([2845,12000,2845]))
        self.assertEqual(ns['stable_system_baseline']([2845,2846,2845]),2846)

if __name__ == '__main__':
    unittest.main()
