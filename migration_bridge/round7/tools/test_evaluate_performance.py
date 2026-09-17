import copy
import unittest
from evaluate_performance import evaluate, BUCKETS, BUDGETS, ORDER


def fixtures(factor=.9):
    rows = [dict(bucket=b, block=k, turn=t, side=ORDER[t],
                 mode='frequency' if ORDER[t]=='left' else 'dynamic',
                 request_seconds=100*(factor if ORDER[t]=='right' else 1),
                 decode_tokens_per_second=10/(factor if ORDER[t]=='right' else 1),
                 output_ids=[1]*BUDGETS[b],prompt_id=b,input_tokens=256)
            for b in BUCKETS for k in range(3) for t in range(4)]
    windows = [dict(bucket=r['bucket'],block=r['block'],turn=r['turn'],formal_eligible=True) for r in rows]
    return dict(status='COMPLETE',observer=False,serial=True,measurements=rows), dict(windows=windows)


class PerformanceDecision(unittest.TestCase):
    def test_complete_gain(self):
        result=evaluate(*fixtures())
        self.assertEqual(result['recommended_dynamic_default_gate'],'PASS')
        self.assertEqual(result['improved_workloads'],3)

    def test_no_benefit_is_legitimate(self):
        self.assertEqual(evaluate(*fixtures(1))['recommended_dynamic_default_gate'],'NO_BENEFIT')

    def test_missing_host_not_no_benefit(self):
        report,_=fixtures()
        self.assertEqual(evaluate(report,{})['recommended_dynamic_default_gate'],'PENDING')

    def test_missing_or_duplicate_cell(self):
        report,host=fixtures();report['measurements'][-1]=copy.deepcopy(report['measurements'][0])
        with self.assertRaisesRegex(ValueError,'COMPLETENESS'):evaluate(report,host)

    def test_wrong_order(self):
        report,host=fixtures();report['measurements'][0]['side']='right'
        with self.assertRaisesRegex(ValueError,'ORDER'):evaluate(report,host)

    def test_median_regression_disqualifies(self):
        report,host=fixtures()
        for r in report['measurements']:
            if r['bucket']=='long' and r['side']=='right':r['request_seconds']=106
        result=evaluate(report,host)
        self.assertEqual(result['improved_workloads'],3)
        self.assertEqual(result['recommended_dynamic_default_gate'],'NO_BENEFIT')

    def test_partial_decode_rates_not_averaged(self):
        report,host=fixtures();report['measurements'][0]['decode_tokens_per_second']=None
        w=evaluate(report,host)['workloads']['short']
        self.assertIsNone(w['throughput_improvement_percent'])
        self.assertTrue(w['meets_3_percent_gain'])

    def test_bad_budget(self):
        report,host=fixtures();report['measurements'][0]['output_ids']=[1]
        with self.assertRaisesRegex(ValueError,'BUDGET'):evaluate(report,host)

    def test_contended_window(self):
        report,host=fixtures();host['windows'][0]['formal_eligible']=False
        self.assertEqual(evaluate(report,host)['status'],'PENDING_HOST_ISOLATION')


if __name__=='__main__':unittest.main()
