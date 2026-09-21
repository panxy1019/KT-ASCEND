import importlib.util
import pathlib
import unittest
P=pathlib.Path(__file__).parents[1]/'runtime/ple_hash.py'
class PLEHashTests(unittest.TestCase):
    def make(self):
        spec=importlib.util.spec_from_file_location('ple_hash',P);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        return m.NGramHasher(ngram_size=3,heads_per_ngram=2,vocab_size=16,eos_token_id=9,
                            multipliers=(3,5,7),head_sizes=(11,13,17,19),head_offsets=(0,11,24,41))
    def test_known_rows(self):
        h=self.make();ids,state=h.hash_tokens([1,2])
        self.assertEqual(ids,[[2,18,24,58],[3,14,33,44]])
        self.assertEqual(state,(1,2))
    def test_eos_resets_following_history_not_current_eos(self):
        h=self.make();ids,_=h.hash_tokens([1,2,9,3])
        self.assertEqual(ids[-1],h.hash_tokens([3])[0][0])
        self.assertNotEqual(ids[-2],h.hash_tokens([9])[0][0])
    def test_chunk_decode_and_request_isolation(self):
        h=self.make();tokens=[1,2,9,3,4,4,0,15]
        want,state=h.hash_tokens(tokens)
        got=[];history=None
        for t in tokens:
            rows,history=h.hash_tokens([t],history);got+=rows
        self.assertEqual((got,history),(want,state))
        a,ha=h.hash_tokens(tokens[:3]);b,hb=h.hash_tokens(tokens[3:],ha)
        self.assertEqual((a+b,hb),(want,state))
        self.assertEqual(h.hash_tokens([1,2]),self.make().hash_tokens([1,2]))
        self.assertEqual(h.hash_tokens([],ha),([],ha))
    def test_bad_token_and_history_rejected(self):
        h=self.make()
        for tokens,history in [([-1],None),([16],None),([True],None),([1],(2,)),([1],(2,99))]:
            with self.subTest(tokens=tokens),self.assertRaises(ValueError):h.hash_tokens(tokens,history)
