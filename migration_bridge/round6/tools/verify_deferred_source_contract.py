#!/usr/bin/env python3
"""Static Round 6 source guard usable without torch or an Ascend runtime."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
CHILD = ROOT / "third_party/sglang/python/sglang/srt/layers/moe/kt_ep_wrapper.py"
PARENT = ROOT / "kt-kernel/python/experts_base.py"


def function_names(tree: ast.AST) -> set[str]:
    return {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}


def main() -> int:
    child_source = CHILD.read_text(encoding="utf-8")
    parent_source = PARENT.read_text(encoding="utf-8")
    child_tree = ast.parse(child_source, filename=str(CHILD))
    parent_tree = ast.parse(parent_source, filename=str(PARENT))

    child_functions = function_names(child_tree)
    parent_functions = function_names(parent_tree)

    required_child = {
        "create_kt_config_from_server_args",
        "layer_needs_kt_wrapper",
        "resolve_layer_max_deferred",
    }
    required_parent = {
        "select_deferred_experts",
        "submit_forward",
        "sync_forward",
    }
    assert required_child <= child_functions, required_child - child_functions
    assert required_parent <= parent_functions, required_parent - parent_functions
    assert "layer_max_deferred = resolve_layer_max_deferred(" in child_source
    assert "max_deferred_experts_per_token=layer_max_deferred" in child_source
    assert "acl.rt.memcpy" not in parent_source
    assert "current_slot = self.layer_idx % KExpertsCPUBuffer.buffer_depth" in parent_source
    assert "incremental = BaseMoEWrapper._layer_has_pending_deferred.get(self.layer_idx - 1, False)" in parent_source
    print("ROUND6_DEFERRED_SOURCE_CONTRACT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
