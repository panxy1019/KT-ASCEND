import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from audit_qwen_dynamic_results import check_raw_rows

class ResultAudit(unittest.TestCase):
    def test_missing_duplicate_or_changed_input_rejected(self):
        expected=[dict(id='a',input_ids=[1,2]),dict(id='b',input_ids=[3,4])]
        rows=[dict(id=r['id'],payload=dict(input_ids=r['input_ids']),http_status=200) for r in expected]
        check_raw_rows(rows,expected)
        for bad in [rows[:1],[rows[0],rows[0]],
                    [dict(id='a',payload=dict(input_ids=[9]),http_status=200),rows[1]]]:
            with self.assertRaises(AssertionError):check_raw_rows(bad,expected)
