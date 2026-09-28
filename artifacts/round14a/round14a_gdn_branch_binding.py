"""One-shot evidence hooks for the exact Round14a GDN model branch.

This module deliberately lives outside ``qwen4_exp.py``.  Round12's promoted
QSA source gate binds that consumer file byte-for-byte; changing it merely to
add diagnostics would create an unrelated QSA implementation identity.  The
model runner calls :func:`gdn_branch_model_ready` after model loading and each
hook writes exactly once, after the corresponding GDN module actually returns.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import time


EXPECTED_QSA_LAYERS = frozenset(range(3, 48, 4))
EXPECTED_GDN_LAYERS = tuple(sorted(set(range(48)) - EXPECTED_QSA_LAYERS))


def _branch(module) -> str:
    return "candidate" if module.round14_gdn_backend_enabled else "baseline"


def _append(path: Path, item: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(item, allow_nan=False, sort_keys=True) + "\n")
        stream.flush()


def _make_hook(path: Path):
    def hook(module, inputs, output):
        del inputs, output
        if getattr(module, "_round14a_gdn_branch_logged", False):
            return
        layer = int(module.layer_id)
        _append(path, {
            "schema": 1,
            "status": "ROUND14A_GDN_BRANCH_HIT",
            "layer": layer,
            "implementation": _branch(module),
            "completed_forward": True,
            "monotonic_ns": time.monotonic_ns(),
        })
        module._round14a_gdn_branch_logged = True
    return hook


def gdn_branch_model_ready(runner) -> None:
    raw = os.environ.get("ROUND14A_GDN_BRANCH_PATH")
    if not raw:
        return
    path = Path(raw)
    if not path.is_absolute() or path.exists():
        raise RuntimeError("Round14a GDN branch evidence path must be fresh and absolute")
    found = {}
    for _name, module in runner.model.named_modules():
        if module.__class__.__name__ != "Qwen4ReferenceGatedDeltaNet":
            continue
        layer = int(module.layer_id)
        if layer in found:
            raise RuntimeError(f"duplicate Round14a GDN layer {layer}")
        if not hasattr(module, "round14_gdn_backend_enabled"):
            raise RuntimeError("Round14a GDN module lacks frozen implementation selector")
        found[layer] = module
    if tuple(sorted(found)) != EXPECTED_GDN_LAYERS:
        raise RuntimeError(
            f"Round14a GDN module topology mismatch: {sorted(found)}"
        )
    hook = _make_hook(path)
    runner._round14a_gdn_branch_hooks = [
        found[layer].register_forward_hook(hook) for layer in EXPECTED_GDN_LAYERS
    ]

