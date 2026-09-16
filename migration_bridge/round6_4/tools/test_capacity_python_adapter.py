import unittest
from unittest.mock import patch, mock_open
from capacity_python_adapter import adapt, main


class CapacityTests(unittest.TestCase):
    def setUp(self):
        self.argv = ["-m", "sglang.launch_server", "--context-length", "512",
                     "--max-total-tokens", "512", "--chunked-prefill-size", "512",
                     "--max-prefill-tokens", "512", "--max-running-requests", "1",
                     "--tp-size", "1", "--disable-cuda-graph", "--kt-method", "LLAMAFILE",
                     "--kt-max-deferred-experts-per-token", "1"]

    def test_only_capacity_changes(self):
        for batch in (1, 2, 4):
            got = adapt(self.argv, batch)
            self.assertEqual(got[-7:], self.argv[-7:])
            self.assertEqual(got[got.index("--max-running-requests")+1], str(batch))
            self.assertEqual(got[got.index("--context-length")+1], "1024")
            self.assertEqual(self.argv[3], "512")

    def test_no_duplicate_or_unexpected_options(self):
        with self.assertRaises(ValueError):
            adapt(self.argv + ["--context-length", "512"], 1)
        with self.assertRaises(ValueError):
            adapt(self.argv, 8)
        self.argv[3] = "1024"
        with self.assertRaises(ValueError):
            adapt(self.argv, 1)

    def test_exec_preserves_virtualenv_program_identity(self):
        with patch("capacity_python_adapter.sys.argv", ["adapter", *self.argv]), \
             patch.dict("os.environ", {"KT_R64_BATCH_CAP":"2", "KT_R64_RESOLVED_COMMAND":"/workspace/results/r64-test/command.json"}), \
             patch("pathlib.Path.open", mock_open()), \
             patch("capacity_python_adapter.os.execv") as execute:
            main()
        executable, argv = execute.call_args.args
        self.assertEqual(executable, "/workspace/results/venv-r6/bin/python")
        self.assertEqual(argv[0], executable)


if __name__ == "__main__":
    unittest.main()
