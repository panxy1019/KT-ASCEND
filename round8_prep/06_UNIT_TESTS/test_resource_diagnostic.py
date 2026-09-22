import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from qwen_resource_diagnostic import cgroup_path

class CgroupPathTest(unittest.TestCase):
    def test_resolves_owned_unified_group(self):
        self.assertEqual(cgroup_path('0::/system.slice/docker-abc.scope\n'),
                         Path('/sys/fs/cgroup/system.slice/docker-abc.scope'))
    def test_rejects_root_and_escape(self):
        for value in ('0::/\n', '0::/../../etc\n', '2:cpu:/x\n'):
            with self.assertRaises(ValueError): cgroup_path(value)

if __name__ == '__main__': unittest.main()
