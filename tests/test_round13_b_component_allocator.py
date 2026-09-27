import ast
from pathlib import Path
import unittest


PROBE = (
    Path(__file__).parents[1]
    / "artifacts"
    / "round13"
    / "round13_b_qsa_component_probe.py"
)


def load_allocator_check():
    tree = ast.parse(PROBE.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "allocator_returned_to_baseline"
    )
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(PROBE), "exec"), namespace)
    return namespace["allocator_returned_to_baseline"]


class Round13BComponentAllocatorTest(unittest.TestCase):
    def test_cleanup_is_compared_with_initialized_runtime_baseline(self):
        allocator_returned_to_baseline = load_allocator_check()
        self.assertTrue(allocator_returned_to_baseline(16896, 2097152, 16896, 2097152))
        self.assertFalse(allocator_returned_to_baseline(16896, 2097152, 33792, 2097152))
        self.assertFalse(allocator_returned_to_baseline(16896, 2097152, 16896, 4194304))

        source = PROBE.read_text()
        seed = source.index("torch.manual_seed(0)")
        baseline = source.index("baseline_allocated =")
        workload = source.index("for layer in LAYERS:")
        self.assertLess(seed, baseline)
        self.assertLess(baseline, workload)
        self.assertIn('"baseline_allocated": baseline_allocated', source)
        self.assertIn('"baseline_reserved": baseline_reserved', source)
        self.assertIn("del future", source)


if __name__ == "__main__":
    unittest.main()
