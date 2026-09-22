from pathlib import Path
import tempfile
import unittest
from run_descriptive_holdout import read_inputs


class HoldoutTests(unittest.TestCase):
    def test_exact_existing_inputs(self):
        folder = Path(__file__).resolve().parents[2]/"round6_3/corpora/frozen"
        value = read_inputs(folder)
        self.assertEqual(len(value["requests"]), 136)
        self.assertEqual(value["token_budget"], 64)

    def test_modified_input_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d)/"hd3.json").write_text('{}')
            with self.assertRaisesRegex(ValueError, "DRIFT"):
                read_inputs(d)


if __name__ == "__main__":
    unittest.main()
