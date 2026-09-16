import unittest
import hashlib
import json
from pathlib import Path
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


class FrozenInputs(unittest.TestCase):
    def test_independent_and_bounded(self):
        p=Path(__file__).resolve().parents[1]/'corpora/r65-r66b-base-protocol-freeze-v1/quality.json'
        j=json.loads(p.read_text())
        self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),'885ccf76386dcfed2867142417300ac69c4401754360f92e6b6e4f179adffd97')
        self.assertEqual(len(j['rows']),288)
        self.assertFalse({r['id'] for r in j['rows']}&set(j['excluded_ids']))
        self.assertEqual(len(j['gsm_exemplars']),8)
        self.assertTrue(all(len(v)==5 for v in j['cmmlu_dev'].values()))
        self.assertTrue(all(len(r['input_ids'])+r['max_new_tokens']<=2048 for r in j['rows']))
        self.assertEqual(sum(r['benchmark']=='gsm8k' for r in j['rows']),32)
        self.assertEqual(sum(r['benchmark']=='cmmlu' for r in j['rows']),256)


if __name__=='__main__':unittest.main()
