import ast
from pathlib import Path
import re
import unittest


SOURCE = Path(__file__).parents[1] / "round11_monitor_owned.py"
TREE = ast.parse(SOURCE.read_text())
NODE = next(
    node for node in TREE.body if isinstance(node, ast.FunctionDef) and node.name == "owned_container_name"
)
SCOPE = {"re": re}
exec(compile(ast.Module([NODE], type_ignores=[]), str(SOURCE), "exec"), SCOPE)


class Round11MonitorOwnershipTests(unittest.TestCase):
    def test_round11_stage_names_are_explicit(self):
        accepted = (
            "kt-round11-t2-baseline-v113",
            "kt-round11-t2-counters-v114",
            "kt-round11-t2-attribution-v115",
            "kt-round11-t3-candidate1-v116",
            "kt-round11-t4-static-v120",
            "kt-round11-t4-dynamic-v121",
            "kt-round11-t5-hotspot-v122",
            "kt-round11-t6-baseline-v130",
            "kt-round11-t6-final-v131",
        )
        self.assertTrue(all(SCOPE["owned_container_name"](name) for name in accepted))

    def test_nearby_and_external_names_are_rejected(self):
        rejected = (
            "mzy-train",
            "kt-round11-t2-baseline-v113-other",
            "kt-round11-t2-random-v113",
            "kt-round11-t7-final-v1",
            "kt-round11-t6-final-vx",
        )
        self.assertTrue(all(not SCOPE["owned_container_name"](name) for name in rejected))


if __name__ == "__main__":
    unittest.main()
