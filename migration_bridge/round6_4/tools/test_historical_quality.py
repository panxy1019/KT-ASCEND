import json
from pathlib import Path
import tempfile
import unittest
from run_historical_quality_smoke import checked_manifest


class HistoricalTests(unittest.TestCase):
    def test_frozen_manifest(self):
        path = Path(__file__).resolve().parents[2]/"round4a4/corpora/d_manifest.json"
        result = checked_manifest(path)
        self.assertEqual(sum(len(x["samples"]) for x in result["benchmarks"].values()),256)

    def test_changed_manifest(self):
        path = Path(__file__).resolve().parents[2]/"round4a4/corpora/d_manifest.json"
        value = json.loads(path.read_text())
        value["extra"] = "modified"
        with tempfile.TemporaryDirectory() as d:
            changed = Path(d)/"manifest.json"
            changed.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError,"DRIFT"):
                checked_manifest(changed)


if __name__ == "__main__":
    unittest.main()
