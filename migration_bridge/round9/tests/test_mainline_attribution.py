import unittest
from migration_bridge.round9.runtime.mainline_attribution import observe, validate_capture
from migration_bridge.round9.tools.analyze_mainline_timeline import interval_union, summarize

class AttributionTests(unittest.TestCase):
    def test_observer_forwards_once_and_preserves_exception_and_result(self):
        calls=[];events=[]
        def original(x):
            calls.append(x)
            if x<0: raise ValueError('original failure')
            return x+1
        wrapped=observe(original,events.append,'cpu_wait',0,lambda:{'mode':'decode'})
        self.assertEqual(wrapped(3),4)
        with self.assertRaisesRegex(ValueError,'original failure'):wrapped(-1)
        self.assertEqual(calls,[3,-1]);self.assertEqual(len(events),2)
        self.assertEqual([e['outcome'] for e in events],['return','error'])
        self.assertTrue(all(e['device_ms'] is None for e in events))

    def test_overlap_union_does_not_add_nested_work(self):
        self.assertEqual(interval_union([(1,9),(2,4),(8,11),(15,18)]),13)
        with self.assertRaises(ValueError):interval_union([(5,4)])

    def test_host_intervals_do_not_prove_device_overlap(self):
        d={'identity':{'request_id':'r'},'events':[dict(category='mainline',phase='span',start_ns=2,end_ns=8,component='GDN',mode='decode',device_ms=3),dict(category='mainline',phase='batch_coverage')],'dropped_events':0}
        r=summarize([d]);self.assertEqual(r['device_overlap'],'UNKNOWN_UNCALIBRATED_CLOCKS')
        self.assertEqual(r['host_union_ns'],6);self.assertIsNone(r['cpu_worker_ns'])
        self.assertEqual(r['decode_spans'],1)
        d['dropped_events']=1
        self.assertEqual(summarize([d])['status'],'PARTIAL_TRUNCATED')

    def test_missing_span_boundary_reports_partial_instead_of_zero(self):
        r=summarize([{'events':[dict(category='mainline',phase='span',start_ns=2,component='GDN')],'dropped_events':0}])
        self.assertEqual(r['invalid_spans'],1);self.assertIsNone(r['host_union_ns'])

    def test_capture_rejects_wrong_bank_layer_layout_and_oversize(self):
        expected={'model_revision':'model-a','bank_sha256':'bank-a','layer':0,'layout':'NZ29','cpu_precision':'MXFP4','npu_precision':'BF16'}
        doc={**expected,'mode':'decode','tokens':1,'hidden':2560,'topk':10,'resident_ids':list(range(32)),
             'request_id':'r9-mainline-capture','intrusive_capture':True,'selected_row':0,
             'original_physical_rows':1,'actual_weight_formats':{'w13_weight':29,'w2_weight':29}}
        validate_capture(doc,expected)
        for key,value in [('bank_sha256','bank-b'),('layer',1),('layout','ND2'),('tokens',33),('resident_ids',list(range(0,512,16))),('request_id','other'),('intrusive_capture',False),('selected_row',1),('original_physical_rows',0),('original_physical_rows',33),('actual_weight_formats',{'w13_weight':2,'w2_weight':29})]:
            with self.assertRaises(ValueError):validate_capture({**doc,key:value},expected)

if __name__=='__main__':unittest.main()

class DefaultOffTests(unittest.TestCase):
    def test_disabled_binding_never_touches_scheduler_or_allocates(self):
        import os
        from unittest.mock import patch
        from migration_bridge.round9.runtime.mainline_attribution import before,after
        with patch.dict(os.environ,{},clear=True):
            before(None,None)
            after(None)

class ScopeTests(unittest.TestCase):
    def test_wrong_precision_capacity_or_dynamic_refused_before_binding(self):
        from migration_bridge.round9.runtime.mainline_attribution import validate_scope
        good=dict(storage='MXFP4',placement='static',capacity=32,workers=128,device='npu',revision='de4b8e4d43b917e7706784d8bb445c9af86a3540')
        validate_scope(good)
        for key,value in [('storage','BF16'),('capacity',64),('placement','request_boundary')]:
            with self.assertRaises(ValueError):validate_scope({**good,key:value})

class ReviewRegressionTests(unittest.TestCase):
    def test_device_setup_failure_cannot_skip_provider(self):
        calls=[];events=[]
        def broken():raise RuntimeError('timing unavailable')
        wrapped=observe(lambda:calls.append(1) or 7,events.append,'x',0,lambda:{},broken)
        self.assertEqual(wrapped(),7);self.assertEqual(calls,[1])
        self.assertTrue(events[0]['observer_errors'])

    def test_device_end_and_sink_failure_do_not_mask_provider_exception(self):
        class Event:
            def record(self):raise RuntimeError('observer end failed')
        class Start:
            def record(self):pass
        def original():raise ValueError('provider failure')
        def sink(row):raise RuntimeError('sink failed')
        errors=[]
        wrapped=observe(original,sink,'x',0,lambda:{},lambda:(Start(),Event()),errors.append)
        with self.assertRaisesRegex(ValueError,'provider failure'):wrapped()
        self.assertTrue(errors)

    def test_empty_or_malformed_trace_cannot_claim_usable_attribution(self):
        self.assertEqual(summarize([])['status'],'EVIDENCE_INCOMPLETE')
        self.assertEqual(summarize([{'events':[],'dropped_events':0}])['status'],'EVIDENCE_INCOMPLETE')
        r=summarize([{'events':[dict(category='mainline',phase='span',start_ns=1,end_ns=2)],'dropped_events':0}])
        self.assertEqual(r['invalid_spans'],1)
        self.assertEqual(r['status'],'EVIDENCE_INCOMPLETE')

class DecodeProbeContractTests(unittest.TestCase):
    def test_one_output_token_cannot_be_requested_as_decode_evidence(self):
        import subprocess,sys,tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as directory:
            result=subprocess.run([sys.executable,str(Path(__file__).with_name('mainline_http_probe.py')), '--rid','r9-mainline-capture','--kind','decode','--max-new-tokens','1','--output',directory+'/result.json'],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('decode requires',result.stderr)
            self.assertFalse(Path(directory,'result.json').exists())
