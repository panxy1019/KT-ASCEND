import ast
from pathlib import Path
import unittest


GUARD = Path(__file__).resolve().parents[1] / "round12a_monitor_owned.py"


def load_owned_container_name():
    tree = ast.parse(GUARD.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "owned_container_name"
    )
    scope = {"re": __import__("re")}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(GUARD), "exec"),
        scope,
    )
    return scope["owned_container_name"]


class Round12aMonitorScopeTests(unittest.TestCase):
    def test_guard_accepts_only_bounded_round12a_owned_names(self):
        owned_container_name = load_owned_container_name()
        accepted = (
            "kt-round12a-tied-offloader-v170",
            "kt-round12a-nonkt-smoke-v171",
            "kt-round12a-kt-a-v172",
            "kt-round12a-nonkt-a-v173",
            "kt-round12a-nonkt-b-v174",
            "kt-round12a-kt-b-v175",
        )
        rejected = (
            "training",
            "kt-round12a-anything-v170",
            "kt-round12a-kt-c-v172",
            "kt-round12a-nonkt-a-v173-extra",
            "kt-round12a-tied-offloader-v170-extra",
            "kt-round13-nonkt-a-v173",
        )
        self.assertTrue(all(owned_container_name(name) for name in accepted))
        self.assertTrue(all(not owned_container_name(name) for name in rejected))


if __name__ == "__main__":
    unittest.main()
