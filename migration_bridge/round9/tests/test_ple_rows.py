import importlib.util
import json
import pathlib
import struct
import tempfile
import unittest
P=pathlib.Path(__file__).parents[1]/'runtime/ple_rows.py'
PREFIX='model.language_model.layers.1.ple.ple_embedding.ngram_embedding.'
class RowTests(unittest.TestCase):
    def test_concurrent_gathers_share_one_descriptor_per_shard(self):
        import concurrent.futures,os,time
        from unittest.mock import patch
        m=self.module();opened=[];original_open=os.open
        def slow_open(*args,**kwargs):
            fd=original_open(*args,**kwargs);opened.append(fd);time.sleep(.02);return fd
        with tempfile.TemporaryDirectory() as td:
            root=pathlib.Path(td);self.fixture(root)
            with m.ShardedBF16Rows(root,expected_parts=2,expected_width=2) as table:
                with patch.object(m.os,'open',slow_open),concurrent.futures.ThreadPoolExecutor(2) as pool:
                    futures=[pool.submit(table.gather_bytes,[0,0]) for _ in range(2)]
                    self.assertEqual([f.result() for f in futures],[bytes(range(4))*2]*2)
                try:self.assertEqual(len(opened),1)
                finally:
                    # The red implementation leaks the descriptor it overwrites.
                    for fd in opened:
                        if fd not in table._fds.values():os.close(fd)
    def module(self):
        s=importlib.util.spec_from_file_location('ple_rows',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
    def fixture(self,root):
        index={}
        for part in range(2):
            name=f'{PREFIX}shard_{part}.weight';file=f'{part}.safetensors'
            payload=bytes(range(part*12,(part+1)*12))
            header=json.dumps({name:{'dtype':'BF16','shape':[3,2],'data_offsets':[0,12]}}).encode()
            (root/file).write_bytes(struct.pack('<Q',len(header))+header+payload);index[name]=file
        (root/'model.safetensors.index.json').write_text(json.dumps({'weight_map':index}))
    def test_original_bytes_order_duplicates_and_boundaries(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as td:
            root=pathlib.Path(td);self.fixture(root)
            with m.ShardedBF16Rows(root,expected_parts=2,expected_width=2) as table:
                self.assertEqual(table.gather_bytes([5,0,3,3,2]),bytes(range(20,24))+bytes(range(4))+bytes(range(12,16))*2+bytes(range(8,12)))
                self.assertEqual(table.gather_bytes([]),b'')
                for ids in ([-1],[6],[True]):
                    with self.assertRaises(ValueError):table.gather_bytes(ids)
            with self.assertRaises(RuntimeError):table.gather_bytes([0])
    def test_missing_part_or_truncated_source_fails(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as td:
            root=pathlib.Path(td);self.fixture(root)
            with self.assertRaises(ValueError):m.ShardedBF16Rows(root,expected_parts=3,expected_width=2)
            (root/'1.safetensors').write_bytes((root/'1.safetensors').read_bytes()[:-1])
            with self.assertRaises(ValueError):m.ShardedBF16Rows(root,expected_parts=2,expected_width=2)
