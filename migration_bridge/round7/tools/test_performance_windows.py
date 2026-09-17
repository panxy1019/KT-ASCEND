import json
import os
from pathlib import Path
import tempfile
import unittest
from audit_performance_windows import capture_window, audit
from test_evaluate_performance import fixtures


class OriginalBoundaries(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.root=Path(self.temporary.name)
        self.directory=self.root/'short-0-0-uniform';self.directory.mkdir()
        self.row=dict(bucket='short',block=0,turn=0,mode='uniform',prompt_id='holdout_000',
                      input_tokens=71,request_seconds=10.,output_ids=[1]*256,decode_tokens_per_second=25.5)
        self.payload=dict(rid='measure-short-0-0')
        self.input=self.directory/'measure-short-0-0-input.json'
        self.raw=self.directory/'measure-short-0-0.json'
        self.input.write_text(json.dumps(dict(payload=self.payload,started_unix=100.)))
        summary=dict(self.row,final_meta=dict(id='measure-short-0-0'))
        self.raw.write_text(json.dumps(dict(payload=self.payload,summary=summary)))
        os.utime(self.raw,(110.1,110.1))
        self.lifecycle=self.directory/'lifecycle.json'
        self.lifecycle.write_text(json.dumps(dict(status='COMPLETE',started_unix=90.,ended_unix=120.,reaping=dict(complete=True))))

    def tearDown(self):self.temporary.cleanup()

    def test_conservative_actual_mtime_not_inferred(self):
        result=capture_window(self.root,self.row)
        self.assertAlmostEqual(result['ended_unix'],110.1)
        self.assertAlmostEqual(result['serialization_and_clock_slack_seconds'],.1)

    def test_copied_file_timestamp_rejected(self):
        os.utime(self.raw,(500.,500.))
        with self.assertRaisesRegex(ValueError,'DURATION'):capture_window(self.root,self.row)

    def test_clock_goes_back(self):
        os.utime(self.raw,(99.,99.))
        with self.assertRaisesRegex(ValueError,'WALLCLOCK'):capture_window(self.root,self.row)

    def test_raw_mismatch(self):
        self.row['request_seconds']=11.
        with self.assertRaisesRegex(ValueError,'SUMMARY'):capture_window(self.root,self.row)

    def test_different_request(self):
        d=json.loads(self.input.read_text());d['payload']['rid']='other';self.input.write_text(json.dumps(d))
        with self.assertRaisesRegex(ValueError,'REQUEST'):capture_window(self.root,self.row)

    def test_unreaped(self):
        d=json.loads(self.lifecycle.read_text());d['reaping']['complete']=False;self.lifecycle.write_text(json.dumps(d))
        with self.assertRaisesRegex(ValueError,'REAPED'):capture_window(self.root,self.row)

    def test_boundary_outside_lifecycle(self):
        d=json.loads(self.lifecycle.read_text());d['ended_unix']=109.;self.lifecycle.write_text(json.dumps(d))
        with self.assertRaisesRegex(ValueError,'LIFECYCLE'):capture_window(self.root,self.row)

    def campaign(self, overlap=False):
        report,_=fixtures()
        for i,row in enumerate(report['measurements']):
            row['mode']='uniform' if row['side']=='left' else 'frequency'
            row['request_seconds']=10.
            start=100. if overlap else 100.+i*20
            d=self.root/f'{row["bucket"]}-{row["block"]}-{row["turn"]}-{row["mode"]}'
            d.mkdir(exist_ok=True);rid=f'measure-{row["bucket"]}-{row["block"]}-{row["turn"]}'
            payload=dict(rid=rid)
            (d/f'{rid}-input.json').write_text(json.dumps(dict(payload=payload,started_unix=start)))
            raw=d/f'{rid}.json'
            raw.write_text(json.dumps(dict(payload=payload,summary=dict(row,final_meta=dict(id=rid)))))
            os.utime(raw,(start+10.1,start+10.1))
            (d/'lifecycle.json').write_text(json.dumps(dict(status='COMPLETE',started_unix=start-5,
                ended_unix=start+15,reaping=dict(complete=True))))
        samples=[dict(time_unix=x,status='EXCLUSIVE',source_sha256='a'*64) for x in range(95,825,5)]
        return report,samples

    def test_complete_independent_sample_audit(self):
        report,samples=self.campaign()
        isolation,_=audit(self.root,report,samples,'a'*64)
        self.assertEqual(isolation['status'],'PASS')
        self.assertEqual(len(isolation['windows']),36)

    def test_overlapping_original_windows_fail(self):
        report,samples=self.campaign(overlap=True)
        with self.assertRaisesRegex(ValueError,'NON_SERIAL'):audit(self.root,report,samples,'a'*64)

    def test_observer_source_mismatch(self):
        report,samples=self.campaign()
        with self.assertRaisesRegex(ValueError,'SOURCE_IDENTITY'):audit(self.root,report,samples,'b'*64)


if __name__=='__main__':unittest.main()
