import json
from pathlib import Path
import tempfile
import unittest
from collect_parallel_probes import collect


class ParallelCollectionTest(unittest.TestCase):
    def fixture(self,base):
        prov=base/'provenance.json'
        prov.write_text(json.dumps(dict(files={'native':'same'},gguf='same',
            cpu_native_sha256='same',bf16_context_port='same')))
        sources=[]
        for threshold in (256,512,1024):
            root=base/str(threshold);root.mkdir()
            (root/'summary.json').write_text(json.dumps(dict(status='PASS',threshold=threshold)))
            (root/'protocol.json').write_text(json.dumps(dict(threshold=threshold,
                dev_sha256='dev',profile_sha256='profile',profile='uniform')))
            for stage in ('process-a','process-b','observer-off'):
                path=root/stage;path.mkdir()
                (path/'lifecycle.json').write_text(json.dumps(dict(status='COMPLETE',reaping={'complete':True})))
            sources.append((threshold,root,prov))
        return sources,prov

    def test_collection(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);sources,prov=self.fixture(base)
            collect(sources,base/'out',prov)
            self.assertTrue((base/'out/threshold-512/summary.json').exists())

    def test_mixed_corpus_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);sources,prov=self.fixture(base)
            p=sources[1][1]/'protocol.json';d=json.loads(p.read_text())
            d['dev_sha256']='different';p.write_text(json.dumps(d))
            with self.assertRaisesRegex(AssertionError,'PROTOCOL_DRIFT'):collect(sources,base/'out',prov)

    def test_unreaped_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);sources,prov=self.fixture(base)
            p=sources[0][1]/'process-a/lifecycle.json'
            p.write_text(json.dumps(dict(status='COMPLETE',reaping={'complete':False})))
            with self.assertRaises(AssertionError):collect(sources,base/'out',prov)
