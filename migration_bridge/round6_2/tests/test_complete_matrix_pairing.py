from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from audit_complete_matrix import grouped_pairs


class CompletePairingTests(unittest.TestCase):
    def test_pairs(self):
        rows = [dict(request_id=rid, kind=kind) for rid in ("a", "b") for kind in ("TARGET", "SERVING_LOOKAHEAD")]
        self.assertEqual(set(grouped_pairs(rows, ["a", "b"])), {"a", "b"})

    def test_total_count_is_not_sufficient(self):
        # Four total forwards + all targets, but a duplicated lookahead for a
        # and no lookahead for b must fail instead of passing aggregate counts.
        rows = [dict(request_id="a", kind="TARGET"), dict(request_id="b", kind="TARGET"),
                dict(request_id="a", kind="SERVING_LOOKAHEAD"), dict(request_id="a", kind="SERVING_LOOKAHEAD")]
        with self.assertRaisesRegex(ValueError, "PER_REQUEST_PAIR"):
            grouped_pairs(rows, ["a", "b"])

    def test_unknown_request(self):
        with self.assertRaisesRegex(ValueError, "REQUEST_SET"):
            grouped_pairs([], ["a"])
