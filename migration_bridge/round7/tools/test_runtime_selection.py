import unittest
from runtime_selection import select_cpu_runtime


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.root='/workspace/c64-dev/round7/build/runtime'
        native=self.root+'/kt_kernel/kt_kernel_ext.cpython-311-aarch64-linux-gnu.so'
        self.cfg=dict(cpu_runtime_root=self.root)
        self.prov=dict(cpu_runtime_root=self.root,cpu_native_sha256='a'*64,
            files={native:dict(sha256='a'*64)})
    def test_default(self):
        self.assertIsNone(select_cpu_runtime({},dict(files={})))
    def test_private(self):
        self.assertEqual(select_cpu_runtime(self.cfg,self.prov),self.root)
    def test_missing_selection(self):
        with self.assertRaisesRegex(ValueError,'NOT_SELECTED'):select_cpu_runtime({},self.prov)
    def test_scope(self):
        for value in ('relative','/tmp/runtime','/workspace/c64-dev/round7/../runtime'):
            with self.assertRaisesRegex(ValueError,'OUTSIDE_OWNED_SCOPE'):
                select_cpu_runtime(dict(cpu_runtime_root=value),self.prov)
    def test_wrong_provenance(self):
        with self.assertRaisesRegex(ValueError,'PROVENANCE_MISMATCH'):
            select_cpu_runtime(self.cfg,{**self.prov,'cpu_runtime_root':self.root+'-wrong'})
    def test_missing_native(self):
        with self.assertRaisesRegex(ValueError,'NOT_FROZEN'):
            select_cpu_runtime(self.cfg,{**self.prov,'files':{}})
    def test_wrong_hash_binding(self):
        with self.assertRaisesRegex(ValueError,'HASH_BINDING'):
            select_cpu_runtime(self.cfg,{**self.prov,'cpu_native_sha256':'b'*64})
