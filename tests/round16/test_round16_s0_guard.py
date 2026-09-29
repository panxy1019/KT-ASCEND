import ast
from pathlib import Path
import re
import unittest


SOURCE = Path(__file__).resolve().parents[2] / "tools" / "monitor_owned_server.py"


def load_owned_container_name():
    tree = ast.parse(SOURCE.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "owned_container_name"
    )
    namespace = {"re": re}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace["owned_container_name"]


class Round16S0GuardTests(unittest.TestCase):
    def test_accepts_only_the_two_owned_s0_performance_name_families(self):
        accepts = load_owned_container_name()
        self.assertTrue(accepts("kt-round16-s0-perf-a-v403"))
        self.assertTrue(accepts("kt-round16-s0-perf-b-v404"))
        self.assertFalse(accepts("kt-round16-s0-perf-c-v403"))
        self.assertFalse(accepts("qwen38-vllm-12"))
        self.assertFalse(accepts("kt-round16-s0-perf-a-v403-extra"))


if __name__ == "__main__":
    unittest.main()
