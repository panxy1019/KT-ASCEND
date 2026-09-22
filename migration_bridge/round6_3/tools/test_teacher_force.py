"""CPU-only protocol tests; not a substitute for A3 scheduler/KV evidence."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SOURCE = Path(os.getenv("KT_R63_TEST_SOURCE", str(Path(__file__).resolve().parents[3] / "third_party/sglang/python/sglang/srt/debug_utils/round63_teacher_force.py")))
spec = importlib.util.spec_from_file_location("r63_teacher_force", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
try:
    import torch
except ImportError:
    torch = None


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


@unittest.skipIf(torch is None, "torch unavailable: tensor adapter needs A3/torch runtime")
class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.manifest = self.root / "manifest.json"
        self.manifest.write_text(json.dumps(dict(version="r63-forced-history-v1", session_id="s",
            requests=[dict(rid="r", input_ids=[10, 11], baseline_output_ids=[20, 21])])) )
        sha = module.hashlib.sha256(self.manifest.read_bytes()).hexdigest()
        self.arm = self.root / "arm"
        self.arm.write_text(sha + "\n")
        self.env = patch.dict(os.environ, dict(KT_R63_FORCE_MANIFEST=str(self.manifest),
            KT_R63_FORCE_MANIFEST_SHA256=sha, KT_R63_FORCE_ARM_FILE=str(self.arm),
            KT_R63_FORCE_TRACE=str(self.root / "trace.jsonl")))
        self.env.start()
        module._ledger, module._trace, module._sequence = None, None, 0

    def tearDown(self):
        if module._trace:
            module._trace.close()
        module._ledger, module._trace, module._sequence = None, None, 0
        self.env.stop()
        self.temp.cleanup()

    def inputs(self):
        logits = torch.zeros(1, 102400)
        logits[0, 42] = 1
        batch = SimpleNamespace(rids=["r"], input_ids=torch.tensor([10, 11]),
            positions=torch.tensor([0, 1]), seq_lens=torch.tensor([2]),
            forward_mode=SimpleNamespace(is_decode=lambda: False),
            sampling_info=SimpleNamespace(is_all_greedy=True))
        return torch.tensor([42]), SimpleNamespace(next_token_logits=logits), batch

    def test_disabled_is_identity(self):
        tokens, logits, batch = self.inputs()
        with patch.dict(os.environ, {"KT_R63_FORCE_MANIFEST": ""}):
            self.assertIs(module.apply_forced_history(tokens, logits, batch), tokens)

    def test_unarmed_is_identity(self):
        self.arm.unlink()
        tokens, logits, batch = self.inputs()
        self.assertIs(module.apply_forced_history(tokens, logits, batch), tokens)

    def test_forcing_preserves_logits_and_consumes_frozen_history(self):
        tokens, logits, batch = self.inputs()
        before = logits.next_token_logits.clone()
        self.assertEqual(module.apply_forced_history(tokens, logits, batch).tolist(), [20])
        self.assertTrue(torch.equal(before, logits.next_token_logits))
        batch.input_ids, batch.positions, batch.seq_lens = torch.tensor([20]), torch.tensor([2]), torch.tensor([3])
        batch.forward_mode.is_decode = lambda: True
        self.assertEqual(module.apply_forced_history(tokens, logits, batch).tolist(), [21])
        records = [json.loads(line) for line in (self.root / "trace.jsonl").read_text().splitlines()]
        self.assertEqual([r["natural_token"] for r in records], [42, 42])
        self.assertEqual([r["sequence"] for r in records], [0, 1])

    def test_nonfinite_rejected(self):
        tokens, logits, batch = self.inputs()
        logits.next_token_logits[0, 10] = float("nan")
        with self.assertRaisesRegex(RuntimeError, "NONFINITE"):
            module.apply_forced_history(tokens, logits, batch)

    def test_hash_drift_rejected(self):
        self.manifest.write_text(self.manifest.read_text() + "\n")
        with self.assertRaisesRegex(RuntimeError, "HASH_MISMATCH"):
            module.apply_forced_history(*self.inputs())

    def test_batch_rejected(self):
        tokens, logits, batch = self.inputs()
        batch.rids = ["r", "other"]
        with self.assertRaisesRegex(RuntimeError, "BATCH1_REQUIRED"):
            module.apply_forced_history(tokens, logits, batch)


if __name__ == "__main__":
    unittest.main()
