from pathlib import Path
import unittest
from run_engineering_oracle import make_manifest, validate_reader, PROMPTS


class OracleManifestTests(unittest.TestCase):
    def setUp(self):
        self.reader = Path(__file__).resolve().parents[3]/"third_party/sglang/python/sglang/srt/debug_utils/round63_teacher_force.py"
        self.payload = make_manifest({p:{"input_ids":[10,11]} for p in PROMPTS}, {p:[20]*64 for p in PROMPTS})

    def test_real_reader_accepts_wire_schema(self):
        validate_reader(self.payload, self.reader)

    def test_missing_version_caught_before_a3_start(self):
        del self.payload["version"]
        with self.assertRaisesRegex(ValueError, "MANIFEST_VERSION"):
            validate_reader(self.payload, self.reader)

    def test_bad_history_count(self):
        self.payload["requests"][0]["baseline_output_ids"].pop()
        with self.assertRaisesRegex(ValueError, "HISTORY_COUNT"):
            validate_reader(self.payload, self.reader)


if __name__ == "__main__":
    unittest.main()
