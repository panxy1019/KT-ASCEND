import json
from pathlib import Path
import struct
import tempfile
import unittest
from migration_bridge.round9.runtime.expert_source import OriginalExpertSource
from migration_bridge.round9.tools.convert_candidate_bank import convert_layer,require_trusted_root,TRUSTED_ROOTS


def source_fixture(root):
    h=32;i=32;e=2
    prefix='model.language_model.layers.0.mlp.experts.'
    blobs=[b''.join(struct.pack('<H',v)*(i*h) for v in (0x3f80,0x4000,0x4040,0x4080)),b''.join(struct.pack('<H',v)*(h*i) for v in (0xbf80,0xc000))]
    keys=[prefix+'gate_up_proj',prefix+'down_proj'];offset=0;header={}
    for k,shape,data in zip(keys,[(e,2*i,h),(e,h,i)],blobs):
        header[k]=dict(dtype='BF16',shape=shape,data_offsets=[offset,offset+len(data)]);offset+=len(data)
    raw=json.dumps(header).encode()
    (root/'weights.safetensors').write_bytes(struct.pack('<Q',len(raw))+raw+b''.join(blobs))
    (root/'model.safetensors.index.json').write_text(json.dumps(dict(weight_map={k:'weights.safetensors' for k in keys})))
    return OriginalExpertSource(root,layers=1,experts=e,hidden=h,intermediate=i)

class CandidateBankTests(unittest.TestCase):
    def test_packed_original_slices_and_conversion(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=source_fixture(root)
            self.assertEqual(source.read(0,1,'up'),struct.pack('<H',0x4080)*1024)
            self.assertEqual(source.read(0,0,'down'),struct.pack('<H',0xbf80)*1024)
            out=root/'bank';report=convert_layer(source,0,[0,1],'Q8_0',out,source_identity={'scope':'TEST_FIXTURE'})
            self.assertEqual(report['status'],'PASS_CONVERSION_ONLY_NOT_NATIVE_QUALIFIED')
            self.assertEqual(report['logical_ids'],[0,1])
            self.assertEqual((out/report['file']).read_bytes()[:4],b'GGUF')
            self.assertEqual(len(report['projection_identities']),6)
            with self.assertRaises(FileExistsError):convert_layer(source,0,[0,1],'Q8_0',out,source_identity={})

    def test_reject_truncation_wrong_dtype_and_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=source_fixture(root)
            for ids in ([1,1],[-1],[2],[]):
                with self.assertRaises(ValueError):convert_layer(source,0,ids,'MXFP4',root/'bad',source_identity={})
            with self.assertRaises(ValueError):source.read(0,True,'gate')
            path=root/'weights.safetensors';path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaises(ValueError):source.read(0,1,'down')

    def test_operational_source_requires_frozen_root_and_readonly_mount(self):
        with self.assertRaises(ValueError):require_trusted_root('/tmp/unverified-copy',readonly=True)
        with self.assertRaises(ValueError):require_trusted_root(TRUSTED_ROOTS[0],readonly=False)
        self.assertEqual(require_trusted_root(TRUSTED_ROOTS[0],readonly=True),TRUSTED_ROOTS[0].resolve())

    def test_insufficient_scratch_rejected_before_output_creation(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=source_fixture(root)
            with patch('migration_bridge.round9.tools.convert_candidate_bank.shutil.disk_usage',return_value=SimpleNamespace(free=0)):
                with self.assertRaises(ValueError):convert_layer(source,0,[0,1],'Q8_0',root/'out',source_identity={})
            self.assertFalse((root/'out').exists())

if __name__=='__main__':unittest.main()
