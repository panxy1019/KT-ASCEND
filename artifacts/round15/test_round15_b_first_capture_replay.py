#!/usr/bin/env python3
"""Regression for the first full-model resident-MoE graph result."""

import ast
from pathlib import Path
import unittest


SOURCE = Path(__file__).with_name("round15_b_resident_graph.py")


class FirstCaptureReplayTests(unittest.TestCase):
    def test_first_candidate_is_produced_by_replay_after_capture(self):
        module = ast.parse(SOURCE.read_text())
        function = next(
            node
            for node in module.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "apply_resident_moe_graph"
        )
        candidate_assignments = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "candidate"
                for target in node.targets
            )
        ]
        self.assertEqual(len(candidate_assignments), 1)
        value = candidate_assignments[0].value
        self.assertIsInstance(value, ast.Call)
        self.assertIsInstance(value.func, ast.Attribute)
        self.assertIsInstance(value.func.value, ast.Name)
        self.assertEqual(value.func.value.id, "graph")
        self.assertEqual(value.func.attr, "replay")
        self.assertEqual(len(value.args), 1)
        self.assertIsInstance(value.args[0], ast.Name)
        self.assertEqual(value.args[0].id, "dispatch_output")


if __name__ == "__main__":
    unittest.main()
