import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from artifacts.round11.build_round11_t5_qsa_component import build


class Round11T5QSAComponentBuilderTests(unittest.TestCase):
    def test_build_freezes_single_vectorized_candidate(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "v144"
            build(output, version=144)
            contract = json.loads((output / "ROUND11_T5_QSA_COMPONENT_CONTRACT.json").read_text())
            self.assertEqual(contract["version"], 144)
            self.assertEqual(contract["candidate"], "QUERY_AWARE_VECTORIZED_SPARSE_GQA")
            self.assertEqual(contract["changed_runtime_files"], ["runtime/qsa.py"])
            self.assertIn(
                "def sparse_gqa_vectorized",
                (output / "runtime/qsa.py").read_text(),
            )
            manifest = json.loads((output / "source-sha256.json").read_text())
            actual = {
                str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in output.rglob("*")
                if path.is_file() and path.name != "source-sha256.json"
            }
            self.assertEqual(actual, manifest)


if __name__ == "__main__":
    unittest.main()
