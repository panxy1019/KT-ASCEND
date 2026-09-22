import unittest
from dynamic_probe import verify_updates
from dynamic_audit import projection_orientation
from cpu_backing_probe import eviction_mask


def update():
    return dict(layer=1,weights=2,mapping=2,mask=2,wrapper_mask=2,prior_generation=1,
        selected=[0,7],weight_identity_exact=True,mask_mapping_wrapper_exact=True,
        checkpoint_source_identity_exact=True)


class Migration(unittest.TestCase):
    def test_consistent_serial_generation(self):
        verify_updates([update()],{1:2})

    def test_mixed_generation_rejected(self):
        x=update();x['mapping']=1
        with self.assertRaises(ValueError):verify_updates([x],{1:2})

    def test_wrong_slot_count_rejected(self):
        with self.assertRaises(AssertionError):verify_updates([update()],{1:8})

    def test_wrong_weight_identity_rejected(self):
        x=update();x['weight_identity_exact']=False
        with self.assertRaises(AssertionError):verify_updates([x],{1:2})

    def test_independent_checkpoint_source_required(self):
        x=update();x['checkpoint_source_identity_exact']=False
        with self.assertRaises(AssertionError):verify_updates([x],{1:2})

    def test_npu_logical_transpose_only(self):
        self.assertTrue(projection_orientation((2048,2816),(2816,2048)))
        self.assertFalse(projection_orientation((2816,2048),(2816,2048)))
        with self.assertRaises(ValueError):projection_orientation((1408,2048),(2816,2048))

    def test_eviction_keeps_budget_and_logical_identity(self):
        result=eviction_mask([0,8,16,24,32,40,48,56],[0,56])
        self.assertEqual(len(result),8)
        self.assertNotIn(0,result);self.assertNotIn(56,result)
        self.assertTrue({8,16,24,32,40,48}<=set(result))
        with self.assertRaises(ValueError):eviction_mask([0,8],[9])
