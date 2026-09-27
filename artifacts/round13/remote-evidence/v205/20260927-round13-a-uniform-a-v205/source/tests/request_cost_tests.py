"""Host-only observer tests; no provider qualification claims."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from migration_bridge.round9.runtime.request_cost_binding import Recorder, Proxy, bind
from migration_bridge.round9.runtime.request_cost_ledger import CostLedger

class ObserverTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.output=Path(self.temp.name)/'cost.jsonl'
        self.rec=Recorder(self.output,dict(runtime_sha256='r',profile_sha256='p'))
        self.req=SimpleNamespace(rid='req1')

    def test_off_does_not_touch_runner(self):
        with patch.dict('os.environ',{},clear=True):
            self.assertFalse(bind(object()))

    def test_two_chunks_and_decode_one_request(self):
        for mode,start,count in [('extend',0,32),('extend',32,5),('decode',None,None)]:
            self.rec.begin(self.req,mode,start,count)
            self.assertEqual(self.rec.wrap(lambda:42,'compute_enqueue')(),42)
            self.rec.end_batch()
        self.rec.finish('finished')
        d=json.loads(self.output.read_text())
        self.assertEqual(d['totals']['prefill_batch']['calls'],2)
        self.assertEqual(d['totals']['decode_batch']['calls'],1)
        self.assertEqual(d['totals']['compute_enqueue']['calls'],3)
        self.assertEqual(d['status'],'COMPLETE_HOST_ACCOUNTING')
        self.assertEqual(sum(r['self_ns'] for r in d['events']),d['totals']['request']['inclusive_ns'])

    def test_provider_error_is_original(self):
        error=RuntimeError('provider')
        def fail():raise error
        self.rec.begin(self.req,'extend',0,32)
        with self.assertRaises(RuntimeError) as ctx:self.rec.wrap(fail,'prepare')()
        self.assertIs(ctx.exception,error)
        with self.assertRaisesRegex(RuntimeError,'incomplete'):self.rec.finish('error')
        self.assertEqual(json.loads(self.output.read_text())['status'],'INVALID_EVIDENCE')

    def test_switch_before_retirement_rejected(self):
        self.rec.begin(self.req,'extend',0,32);self.rec.end_batch()
        with self.assertRaisesRegex(RuntimeError,'switched'):
            self.rec.begin(SimpleNamespace(rid='req2'),'extend',0,32)

    def test_cancel_then_reuse_isolated(self):
        self.rec.begin(self.req,'extend',0,32);self.rec.finish('cancelled')
        self.rec.begin(SimpleNamespace(rid='req2'),'extend',0,17);self.rec.finish('finished')
        rows=[json.loads(x) for x in self.output.read_text().splitlines()]
        self.assertEqual([x['identity']['request_id'] for x in rows],['req1','req2'])
        self.assertEqual([x['termination'] for x in rows],['cancelled','finished'])
        self.assertTrue(all(x['totals']['prefill_batch']['calls']==1 for x in rows))

    def test_slotted_facade_proxy_preserves_identity_and_result(self):
        class Inner:
            __slots__=('copied_bytes',)
            def __init__(self):self.copied_bytes=0
            def copy_experts(self,layer,moves):self.copied_bytes+=len(moves)*10;return moves
        inner=Inner();moves=((0,3),(1,5))
        proxy=Proxy(inner,'copy_experts',inner.copy_experts)
        self.assertIs(proxy.copy_experts(0,moves),moves)
        self.assertEqual(proxy.copied_bytes,20)

    def test_output_must_be_fresh(self):
        self.output.touch()
        with self.assertRaises(FileExistsError):Recorder(self.output,{})

    def test_bytes_and_group_metadata(self):
        self.rec.begin(self.req,'extend',0,17)
        with self.rec.ledger.span('prepare',layer=3,chunk=(0,17),group=7,expert_count=10) as row:
            row['logical_bytes']=98304000
        self.rec.finish('finished')
        d=json.loads(self.output.read_text())
        self.assertEqual(d['totals']['prepare']['logical_bytes'],98304000)
        self.assertEqual(d['events'][0]['group'],7)

if __name__=='__main__':unittest.main()
