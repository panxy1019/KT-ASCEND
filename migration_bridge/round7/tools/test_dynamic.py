import unittest
from dynamic_probe import verify_updates


def update():
    return dict(layer=1,weights=2,mapping=2,mask=2,wrapper_mask=2,prior_generation=1,
        selected=[0,7],weight_identity_exact=True,mask_mapping_wrapper_exact=True)


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
