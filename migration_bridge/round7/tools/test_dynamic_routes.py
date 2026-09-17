import unittest
from dynamic_route_summary import summarize_decode,correlate_updates


def rows(mode='ForwardMode.DECODE',forward=4):
    return [dict(layer=l,tokens=1,assignments=6,npu_hits=2,cpu_hits=4,
        total_weight=1.,cpu_weight=.6,initial_static_npu_hits=1,
        initial_static_cpu_weight=.8,fully_npu_row_flags=[False],
        forward_context=dict(epoch=1,forward_id=forward,mode=mode)) for l in range(1,27)]


class Decode(unittest.TestCase):
    def test_prefill_and_unused_tail_excluded(self):
        r=summarize_decode(rows()+rows('ForwardMode.EXTEND',3)+rows(forward=5),{(1,3),(1,4)})
        self.assertEqual(r['assignments'],156)
        self.assertEqual(r['model_token_rows'],1)
        self.assertAlmostEqual(r['npu_hit_rate'],1/3)
        self.assertAlmostEqual(r['initial_static_npu_hit_rate'],1/6)
        self.assertAlmostEqual(r['cpu_weight_mass'],.6)

    def test_missing_layer_rejected(self):
        with self.assertRaises(ValueError):summarize_decode(rows()[:-1],{(1,4)})

    def test_duplicate_layer_rejected(self):
        with self.assertRaises(ValueError):summarize_decode(rows()+rows()[:1],{(1,4)})

    def test_prefill_is_not_hybrid_decode(self):
        with self.assertRaises(ValueError):summarize_decode(rows('ForwardMode.EXTEND'),{(1,4)})

    def test_actual_updates_not_expected_threshold_count(self):
        request=[dict(id='short',started_unix=1,ended_unix=2),
                 dict(id='long',started_unix=3,ended_unix=5)]
        event=dict(time_unix=4,layer=1,selected=[1,2],previous_resident=[0,1],migration_bytes=64)
        result=correlate_updates([event],request)
        self.assertEqual(result['actual_trigger_rate'],.5)
        self.assertEqual(result['changed_resident_experts'],1)
        self.assertEqual(result['copied_expert_slots'],2)
        self.assertEqual(result['migration_bytes'],64)
        event['time_unix']=6
        with self.assertRaises(ValueError):correlate_updates([event],request)
