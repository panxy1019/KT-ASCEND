"""Stdlib validation of immutable pre-output bindings and quality independence."""
import hashlib
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
FROZEN=ROOT/'corpora/frozen-v1'


class FreezeTests(unittest.TestCase):
    def test_all_bindings(self):
        manifest=json.loads((FROZEN/'FREEZE_MANIFEST.json').read_text())
        for name,digest in manifest['files'].items():
            self.assertEqual(hashlib.sha256((FROZEN/name).read_bytes()).hexdigest(),digest)

    def test_corpora_disjoint_and_complete(self):
        ids=set();texts=set();tokens=set()
        for name in ('qd6','hd6'):
            corpus=json.loads((FROZEN/(name+'.json')).read_text())
            self.assertEqual(corpus['token_budget'],64)
            self.assertEqual(len(corpus['requests']),12)
            families={}
            for row in corpus['requests']:
                self.assertNotIn(row['rid'],ids);ids.add(row['rid'])
                self.assertNotIn(row['text'],texts);texts.add(row['text'])
                key=tuple(row['input_ids'])
                self.assertNotIn(key,tokens);tokens.add(key)
                families[row['family']]=families.get(row['family'],0)+1
            self.assertEqual(sorted(families.values()),[2]*6)

    def test_quality_screen_not_deferred_contract(self):
        q=json.loads((FROZEN/'quality.json').read_text())
        self.assertFalse(q['population_noninferiority_claim'])
        self.assertEqual(q['margin_absolute'],0.)
        self.assertEqual(sum(r['benchmark']=='gsm8k' for r in q['rows']),32)
        self.assertEqual(sum(r['benchmark']=='cmmlu' for r in q['rows']),256)
        self.assertFalse({r['id'] for r in q['rows']}&set(q['excluded_old_quality_ids']))
        self.assertEqual(len({r['subject'] for r in q['rows'] if r['benchmark']=='cmmlu'}),67)

    def test_contract_does_not_refit_old_bound(self):
        c=json.loads((FROZEN/'CONTRACT_V1.json').read_text())
        self.assertEqual(c['expert_rel_l2_max'],.01)
        self.assertEqual(c['historical_round6_5'],'BLOCKED_UNCHANGED')
        self.assertEqual(c['model_distribution'],'DIAGNOSTIC_ONLY_NO_REFITTED_GLOBAL_MAX_BOUND')
        self.assertFalse(c['candidate_frozen'])
        self.assertEqual(c['deferred'],0)
        self.assertFalse(c['dynamic'])


if __name__=='__main__':unittest.main()
