from pathlib import Path
import unittest


class Round17ProfileRunScriptTests(unittest.TestCase):
    def test_script_has_fresh_identity_no_retry_and_owned_cleanup(self):
        script = (
            Path(__file__).resolve().parents[2]
            / "tools/round17/run_profile_block.sh"
        ).read_text()
        self.assertIn("[[ ! -e \"$EVID\" ]]", script)
        self.assertIn("automatic retry", script.lower())
        self.assertIn("docker rm \"$container_id\"", script)
        self.assertIn("ROUND16_A1_ADMISSION_SHA256", script)
        self.assertIn("result-compatible.json", script)
        self.assertIn("validated_at_epoch", script)
        self.assertIn("open(output_path, \"x\"", script)
        self.assertIn("chmod -R a+rX /out", script)
        self.assertIn("npu-smi info -t proc-mem -i 6 -c 0", script)


if __name__ == "__main__":
    unittest.main()
