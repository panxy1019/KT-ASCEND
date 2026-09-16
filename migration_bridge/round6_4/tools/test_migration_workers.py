import unittest
from capacity_python_adapter import adapt_workers
from owned_server import OwnedServer


class Workers(unittest.TestCase):
    def test_only_worker_flag_changes(self):
        argv=['-m','sglang.launch_server','--kt-cpuinfer','16','--context-length','512']
        changed=adapt_workers(argv,32)
        self.assertEqual(changed,argv[:3]+['32']+argv[4:])
        self.assertEqual(argv[3],'16')

    def test_invalid_or_unfrozen_rejected(self):
        with self.assertRaises(ValueError):
            adapt_workers(['--kt-cpuinfer','16'],64)
        with self.assertRaises(ValueError):
            adapt_workers(['--kt-cpuinfer','8'],32)
        with self.assertRaises(ValueError):
            OwnedServer('/workspace/results/r64-test','D1',cpu_workers=64)


if __name__ == '__main__':
    unittest.main()
