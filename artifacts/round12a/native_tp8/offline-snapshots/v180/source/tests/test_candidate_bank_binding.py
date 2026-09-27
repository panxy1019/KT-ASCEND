import hashlib,json,os,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
from migration_bridge.round9.runtime import candidate_bank as bank
class BankBindingTests(unittest.TestCase):
 def test_actual_type_hash_and_complete_bank_set(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);profile=root/'profile.json';profile.write_text(json.dumps(dict(workers=128,storage='Q8_0',placement='static',numa_nodes=[2,3,6,7])))
   rows=[]
   for layer in range(48):
    name=f'layer-{layer:02d}.gguf';(root/name).write_bytes(bytes([layer]));rows.append(dict(file=name,layer=layer,storage='Q8_0',source_identity={},bytes=1,sha256=bank.digest(root/name)))
   catalog=root/'catalog.json';catalog.write_text(json.dumps(dict(source_identity={},banks=rows)))
   tensor_type=[8]
   def reader(path):
    layer=int(Path(path).stem.split('-')[1]);return NS(tensors=[NS(name=f'blk.{layer}.ffn_{p}_exps.weight',tensor_type=tensor_type[0],shape=NS(tolist=lambda p=p:[640,2560,512] if p=='down' else [2560,640,512])) for p in ('gate','up','down')])
   with patch.dict('sys.modules',{'gguf':NS(GGUFReader=reader)}),patch.object(bank,'source_identity',return_value={}),patch.object(bank.os,'statvfs',return_value=NS(f_flag=os.ST_RDONLY)):
    receipt=bank.verify_catalog(profile,root,root,root/'identity');self.assertEqual(len(receipt['banks']),48)
    tensor_type[0]=39
    with self.assertRaisesRegex(RuntimeError,'actual bank type'):bank.verify_catalog(profile,root,root,root/'identity')
    tensor_type[0]=8;(root/'layer-00.gguf').write_bytes(b'x')
    with self.assertRaisesRegex(RuntimeError,'SHA'):bank.verify_catalog(profile,root,root,root/'identity')
    (root/'layer-00.gguf').write_bytes(b'\0');(root/'extra.gguf').write_bytes(b'x')
    with self.assertRaisesRegex(RuntimeError,'unlisted'):bank.verify_catalog(profile,root,root,root/'identity')
    (root/'extra.gguf').unlink();rows.pop();catalog.write_text(json.dumps(dict(source_identity={},banks=rows)))
    with self.assertRaisesRegex(RuntimeError,'incomplete'):bank.verify_catalog(profile,root,root,root/'identity')
if __name__=='__main__':unittest.main()
