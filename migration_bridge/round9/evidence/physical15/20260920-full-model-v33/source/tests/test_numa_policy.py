import importlib.util
from pathlib import Path
import unittest
spec=importlib.util.spec_from_file_location('numa_policy',Path(__file__).resolve().parents[1]/'tools/numa_policy.py')
api=importlib.util.module_from_spec(spec);spec.loader.exec_module(api)

class NumaPolicyTests(unittest.TestCase):
    def node(self, free, inactive=120, dirty=0):
        fields={'MemFree':free,'Inactive(file)':inactive,'Dirty':dirty,'Writeback':0,'Shmem':0,'Mapped':0}
        return '\n'.join(f'Node 2 {k}: {int(v*1024**2)} kB' for k,v in fields.items())
    def test_current_mxfp4_passes_but_other_banks_keep_floor(self):
        self.assertEqual(api.evaluate_nodes({2:self.node(79.4)},'MXFP4')['free_floor_gib'],48)
        for storage in ('BF16','Q8_0'):
            with self.assertRaises(RuntimeError):api.evaluate_nodes({2:self.node(79.4)},storage)
    def test_free_floor_cannot_be_replaced_by_large_cache(self):
        with self.assertRaises(RuntimeError):api.evaluate_nodes({2:self.node(47)},'MXFP4')
    def test_dirty_cache_does_not_count_as_clean_headroom(self):
        with self.assertRaises(RuntimeError):api.evaluate_nodes({2:self.node(48,120,120)},'MXFP4')
    def test_unknown_storage_fails(self):
        with self.assertRaises(ValueError):api.evaluate_nodes({2:self.node(100)},'Q4')
