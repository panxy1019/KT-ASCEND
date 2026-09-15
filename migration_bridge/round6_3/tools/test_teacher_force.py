"""CPU-only protocol tests; not a substitute for A3 scheduler/KV evidence."""
import importlib.util
from pathlib import Path
import unittest

SOURCE = Path(__file__).resolve().parents[3] / "third_party/sglang/python/sglang/srt/debug_utils/round63_teacher_force.py"
spec = importlib.util.spec_from_file_location("r63_teacher_force", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class HistoryTests(unittest.TestCase):
    def ledger(self):
        return module.HistoryLedger(dict(version="r63-forced-history-v1", session_id="test",
            requests=[dict(rid="r", input_ids=[10, 11], baseline_output_ids=[20, 21])]))

    def test_prefill_decode_lookahead(self):
        ledger = self.ledger()
        a = ledger.observe("r", [10, 11], [0, 1], 2, False, 42)
        b = ledger.observe("r", [20], [2], 3, True, 43)
        c = ledger.observe("r", [21], [3], 4, True, 44)
        self.assertEqual([a["returned_token"], b["returned_token"], c["returned_token"]], [20, 21, 44])
        self.assertEqual(c["classification"], "LOOKAHEAD")
        self.assertEqual(b["history_sha256"], module.digest([10, 11, 20]))
        with self.assertRaisesRegex(ValueError, "EXTRA_FORWARD"):
            ledger.observe("r", [44], [4], 5, True, 45)

    def test_unknown_rid(self):
        with self.assertRaisesRegex(ValueError, "UNKNOWN_RID"):
            self.ledger().observe("other", [10, 11], [0, 1], 2, False, 42)

    def test_wrong_input_position_length(self):
        for ids, pos, length in [([11, 10], [0, 1], 2), ([10, 11], [1, 2], 2), ([10, 11], [0, 1], 3)]:
            with self.subTest(ids=ids, pos=pos, length=length):
                with self.assertRaisesRegex(ValueError, "CONSUMED_HISTORY"):
                    self.ledger().observe("r", ids, pos, length, False, 42)

    def test_duplicate_prefill(self):
        ledger = self.ledger()
        ledger.observe("r", [10, 11], [0, 1], 2, False, 42)
        with self.assertRaisesRegex(ValueError, "MODE_OR_RESTART"):
            ledger.observe("r", [10, 11], [0, 1], 2, False, 42)

    def test_missing_step_or_free_running_input(self):
        for token, pos, length in [(21, 3, 4), (42, 2, 3)]:
            ledger = self.ledger()
            ledger.observe("r", [10, 11], [0, 1], 2, False, 42)
            with self.assertRaisesRegex(ValueError, "CONSUMED_HISTORY"):
                ledger.observe("r", [token], [pos], length, True, 43)

    def test_bad_tokens_and_duplicates(self):
        for tokens in [[], [-1], [102400], [True], [1.0]]:
            with self.assertRaisesRegex(ValueError, "TOKEN_LIST"):
                module.HistoryLedger(dict(version="r63-forced-history-v1", session_id="test",
                    requests=[dict(rid="r", input_ids=[1], baseline_output_ids=tokens)]))
        r = dict(rid="r", input_ids=[1], baseline_output_ids=[2])
        with self.assertRaisesRegex(ValueError, "DUPLICATE"):
            module.HistoryLedger(dict(version="r63-forced-history-v1", session_id="test", requests=[r, r]))

    def test_independent_ledgers_do_not_share_state(self):
        a, b = self.ledger(), self.ledger()
        a.observe("r", [10, 11], [0, 1], 2, False, 42)
        self.assertEqual(b.steps, {})


if __name__ == "__main__":
    unittest.main()
