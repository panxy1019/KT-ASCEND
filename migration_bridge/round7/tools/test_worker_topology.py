import unittest
from worker_topology import topology, output_root,A3_CPUSET,A3_CPUS,cpu_ids


class WorkerTopologyTest(unittest.TestCase):
    def test_a3_explicit_physical_core_profile(self):
        cfg=dict(workers=64,cpu_set=A3_CPUSET,numa_nodes=[0,1],topology_profile='a3-smt-c64-v1')
        self.assertEqual(topology(cfg,False),(A3_CPUSET,[0,1]))
        self.assertEqual(len(set(A3_CPUS)),64)
        for patch in (dict(cpu_set='0-63'),dict(workers=32),dict(numa_nodes=[4,5])):
            with self.assertRaises(ValueError):topology(dict(cfg,**patch),False)

    def test_cpulist_parser(self):
        self.assertEqual(cpu_ids('0-3,8,10-11'),{0,1,2,3,8,10,11})

    def test_original(self):
        self.assertEqual(topology(dict(workers=64,cpu_set='128-191'),False),('128-191',[4,5]))

    def test_relocations(self):
        for cpus,nodes in [('0-63',[0,1]),('64-127',[2,3])]:
            self.assertEqual(topology(dict(workers=64,cpu_set=cpus,numa_nodes=nodes),False),(cpus,nodes))

    def test_reject_changes(self):
        for c in [dict(workers=32,cpu_set='0-63',numa_nodes=[0,1]),
                  dict(workers=64,cpu_set='0-63'),dict(workers=64,cpu_set='64-127',numa_nodes=[4,5])]:
            with self.assertRaises(ValueError):topology(c,False)

    def test_output(self):
        output_root('/workspace/parallel-worker/test')
        for p in ['/tmp/test','relative','/workspace/parallel-worker/../test']:
            with self.assertRaises(ValueError):output_root(p)
