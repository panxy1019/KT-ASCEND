import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from qwen_dynamic_evaluation import quality_payload, heldout_payload, validate_entry


class DynamicEvaluation(unittest.TestCase):
    def test_frozen_quality_protocol_preserved(self):
        row=dict(id='x',input_ids=[11,22],stop=['\n'],candidate_ids=[1,2,3,4])
        p=quality_payload(row)
        self.assertEqual(p['input_ids'],[11,22])
        self.assertEqual(p['sampling_params'],dict(temperature=0,max_new_tokens=1,ignore_eos=False,stop=['\n']))
        self.assertEqual(p['token_ids_logprob'],[1,2,3,4])
        self.assertEqual(p['logprob_start_len'],-1)
    def test_heldout_keeps_full_history(self):
        p=heldout_payload(dict(id='h',input_ids=list(range(3000))))
        self.assertEqual(len(p['input_ids']),3000)
        self.assertEqual(p['sampling_params']['max_new_tokens'],16)
    def test_unfrozen_or_unclosed_entry_rejected(self):
        for entry in [{},dict(local_commit='a'*40,system_gate='PENDING'),
                      dict(local_commit='short',system_gate='PASS')]:
            with self.assertRaises(AssertionError):validate_entry(entry)
        validate_entry(dict(local_commit='a'*40,system_gate='PASS',candidate_sha256='b'*64))
