import unittest
from freeze_fast_quality import select, observed_pass


class FastQualityTests(unittest.TestCase):
    def test_granularity(self):
        self.assertTrue(observed_pass(10,10,32))
        self.assertFalse(observed_pass(10,9,32))
        self.assertTrue(observed_pass(100,95,256))
        self.assertFalse(observed_pass(100,94,256))

    def test_selection(self):
        rows = [dict(id=f'g{i}',benchmark='gsm8k') for i in range(100)]
        rows += [dict(id=f'c{s}-{i}',benchmark='cmmlu',subject=str(s)) for s in range(67) for i in range(10)]
        result = select(rows)
        self.assertEqual(result,select(list(reversed(rows))))
        self.assertEqual(len(result),288)
        self.assertEqual(len({r['subject'] for r in result if r['benchmark']=='cmmlu'}),67)


if __name__ == '__main__':
    unittest.main()
