import json
from pathlib import Path
import unittest

from run_d1_runtime import validate_entry, checked
from run_compact_controls import restart_references, S6_SHA


class EntryTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1]
        self.contract = json.loads((root/'APPROXIMATE_ACCELERATION_CONTRACT.json').read_text())
        self.corpus = checked(root.parent/'round6/corpora/s6.json')

    def test_runtime_does_not_require_quality_finished(self):
        self.assertFalse(self.contract['quality']['primary_quality_execution_allowed'])
        validate_entry(self.corpus, self.contract)

    def test_old_candidate_not_relabelled(self):
        self.contract['version'] = 'round6.3-v2-candidate'
        with self.assertRaises(ValueError):
            validate_entry(self.corpus, self.contract)

    def test_margin_not_changed(self):
        self.contract['quality']['absolute_noninferiority_margin'] = .03
        with self.assertRaises(ValueError):
            validate_entry(self.corpus, self.contract)

    def test_reduced_corpus_not_allowed(self):
        self.corpus['prompts'].pop()
        with self.assertRaises(ValueError):
            validate_entry(self.corpus, self.contract)

    def test_restart_complete_and_binding(self):
        selected = {p['id']:p for p in self.corpus['prompts']}
        report = dict(status='RUNTIME_PASS', profile='D1', binding={'source':'frozen'},
            corpus_sha256=S6_SHA, kind='same_process', request_count=156,
            rows=[dict(prompt_id=pid, reference64=[1]*64) for pid in selected])
        self.assertEqual(len(restart_references(report, 'D1', {'source':'frozen'}, selected)), 12)
        with self.assertRaisesRegex(ValueError, 'DRIFT'):
            restart_references(report, 'D1', {'source':'changed'}, selected)
        report['rows'].pop()
        with self.assertRaisesRegex(ValueError, 'INCOMPLETE'):
            restart_references(report, 'D1', {'source':'frozen'}, selected)


if __name__ == '__main__':
    unittest.main()
