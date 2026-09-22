import json
from pathlib import Path
import tempfile
import unittest
from run_primary_quality import frozen_inputs, sha


class PrimaryFreezeTests(unittest.TestCase):
    def test_preparation_cannot_execute(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/"protocol.json"
            p.write_text(json.dumps(dict(status="PROTOCOL_PREPARATION", primary_execution_allowed=False)))
            with self.assertRaisesRegex(ValueError, "NOT_FROZEN"):
                frozen_inputs(p, Path(d)/"missing.json")

    def test_margin_or_batch_cannot_be_changed(self):
        for margin, batch in ((.03, 1), (.02, 2)):
            with tempfile.TemporaryDirectory() as d:
                p = Path(d)/"protocol.json"
                p.write_text(json.dumps(dict(status="FROZEN_BEFORE_PRIMARY_OUTPUT", primary_execution_allowed=True,
                                            margin_absolute=margin, primary_batch=batch)))
                with self.assertRaisesRegex(ValueError, "SCOPE_DRIFT"):
                    frozen_inputs(p, Path(d)/"missing.json")

    def test_partial_sample_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p, inputs = Path(d)/"protocol.json", Path(d)/"inputs.json"
            inputs.write_text(json.dumps(dict(rows=[])))
            p.write_text(json.dumps(dict(status="FROZEN_BEFORE_PRIMARY_OUTPUT", primary_execution_allowed=True,
                margin_absolute=.02, primary_batch=1, primary_input_sha256=sha(inputs), evaluator_sha256={})))
            with self.assertRaisesRegex(ValueError, "SAMPLE_DRIFT"):
                frozen_inputs(p, inputs)


if __name__ == "__main__":
    unittest.main()
