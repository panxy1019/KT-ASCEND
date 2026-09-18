import unittest
from performance_screen import selected_buckets


class RecoveryScope(unittest.TestCase):
    def test_whole_workload_recovery(self):
        self.assertEqual(selected_buckets('medium,long',1),['medium','long'])

    def test_no_reordering_duplicates_or_unknown_workload(self):
        for value in ('long,medium','short,short','unknown',''):
            with self.assertRaises(ValueError):selected_buckets(value,1)

    def test_formal_cannot_omit_workload(self):
        with self.assertRaises(ValueError):selected_buckets('medium,long',3)
        self.assertEqual(len(selected_buckets('short,medium,long',3)),3)
