import ast
from pathlib import Path
import re
import unittest

source = Path(__file__).parents[1] / 'tools/monitor_owned_server.py'
tree = ast.parse(source.read_text())
nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('validate_owned_limit','parse_physical15','hbm_budget_exceeded','stop_owned','stable_system_baseline','owned_container_name','owned_process_snapshot','owned_memory_cgroup','owned_container_finished')]
ns = {'re': re}
exec(compile(ast.Module(nodes, type_ignores=[]), str(source), 'exec'), ns)

class MonitorOwnershipTests(unittest.TestCase):
    def test_candidate_scope_is_narrow(self):
        for name in ('kt-round9-full-model-v7','kt-round9-c128-expert-v1','kt-round9-layerwise-v2','kt-round9-ple-v1'):
            self.assertTrue(ns['owned_container_name'](name))
        for name in ('mzy-train','kt-r7-a3-return-v1','kt-round9-c128-expert-v1-other',
                     'kt-round9-layerwise-v2-other','kt-round9-layerwise-vx'):
            self.assertFalse(ns['owned_container_name'](name))

class MemoryDomainTests(unittest.TestCase):
    def test_hybrid_host_uses_explicit_v1_memory_controller(self):
        self.assertEqual(ns['owned_memory_cgroup'](['12:memory:/docker/abc123','0::/'],'abc123'),('/sys/fs/cgroup/memory','/docker/abc123'))

    def test_failed_exit_snapshot_cannot_disable_fail_closed_stop(self):
        from unittest.mock import Mock
        ns.update(command=Mock(side_effect=OSError('docker unavailable')),json=__import__('json'))
        self.assertFalse(ns['owned_container_finished']('kt-round9-full-model-v10'))

    def test_root_cgroup_during_exit_is_not_owned_memory(self):
        with self.assertRaisesRegex(RuntimeError,'admitted container'):
            ns['owned_memory_cgroup'](['5:memory:/'],'abc123')
        with self.assertRaises(RuntimeError):ns['owned_memory_cgroup'](['0::/system.slice/docker-other.scope'],'abc123')
        self.assertEqual(ns['owned_memory_cgroup'](['5:memory:/docker/abc123'],'abc123'),('/sys/fs/cgroup/memory','/docker/abc123'))
        self.assertEqual(ns['owned_memory_cgroup'](['0::/system.slice/docker-abc123.scope'],'abc123'),('/sys/fs/cgroup','/system.slice/docker-abc123.scope'))

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

class CandidateBudgetTests(unittest.TestCase):
    def test_exclusive_budget_cannot_relax_shared_guard(self):
        ns['validate_owned_limit'](45056,'exclusive')
        for limit,mode in ((45057,'exclusive'),(45056,'shared'),(0,'exclusive')):
            with self.assertRaises(ValueError):ns['validate_owned_limit'](limit,mode)
        self.assertFalse(ns['hbm_budget_exceeded'](47000,44000,0,2880,45056))
        self.assertTrue(ns['hbm_budget_exceeded'](48000,45057,0,2880,45056))

    def test_expert_budget_is_tighter(self):
        self.assertTrue(ns['hbm_budget_exceeded'](8000,4097,0,2880,4096))
        self.assertFalse(ns['hbm_budget_exceeded'](7000,4000,0,2880,4096))

class MonitorExitRaceTests(unittest.TestCase):
    def test_exit_between_inspect_and_top_is_normal(self):
        import subprocess,json
        from unittest.mock import Mock
        ns.update(subprocess=subprocess,json=json,
            command=Mock(side_effect=[subprocess.CalledProcessError(1,['docker','top']),
                json.dumps([{'State':{'Running':False,'Status':'exited'}}])]))
        self.assertIsNone(ns['owned_process_snapshot']('kt-round9-c128-expert-v1',False))
    def test_live_top_failure_still_fails_closed(self):
        import subprocess,json
        from unittest.mock import Mock
        ns.update(subprocess=subprocess,json=json,
            command=Mock(side_effect=[subprocess.CalledProcessError(1,['docker','top']),
                json.dumps([{'State':{'Running':True,'Status':'running'}}])]))
        with self.assertRaises(subprocess.CalledProcessError):
            ns['owned_process_snapshot']('kt-round9-c128-expert-v1',False)

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
