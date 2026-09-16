import unittest
from run_base_protocol import score_letters


class LetterScoring(unittest.TestCase):
    def test_identity_not_response_order(self):
        letter,values=score_letters([[-3,13],[-1,11],[-4,10],[-2,12]],[10,11,12,13])
        self.assertEqual(letter,'B');self.assertEqual(values,[-4,-1,-2,-3])
    def test_fixed_tie_order(self):
        self.assertEqual(score_letters([[0,x] for x in [13,12,11,10]],[10,11,12,13])[0],'A')
    def test_missing_and_duplicate_fail(self):
        for data in ([[0,10],[0,11],[0,12]],[[0,10],[0,11],[0,12],[0,12]]):
            with self.assertRaises(ValueError):score_letters(data,[10,11,12,13])
    def test_nonfinite_fails(self):
        with self.assertRaises(ValueError):score_letters([[float('nan'),10],[0,11],[0,12],[0,13]],[10,11,12,13])


if __name__=='__main__':unittest.main()
