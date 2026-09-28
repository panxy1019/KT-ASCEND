import json
from pathlib import Path
import tempfile
import unittest

from round14a_resident_model_driver import validate_physical


class Round14aResidentModelReceiptTest(unittest.TestCase):
    def test_uniform_q32_receipt_uses_frozen_summary_total_bytes(self):
        receipt = {
            "schema": 1,
            "status": "RECORDED_Q32_PHYSICAL_NOT_F7_QUALIFIED",
            "summary": {
                "schema": 1, "layers": 48, "resident_capacity": 32,
                "per_layer_bytes": 314572800, "total_bytes": 15099494400,
            },
            "layers": [{} for _ in range(48)],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "q32-physical.json"
            path.write_text(json.dumps(receipt))
            self.assertEqual(
                validate_physical(path, "baseline", Path(directory) / "unused.json"),
                receipt,
            )


if __name__ == "__main__":
    unittest.main()
