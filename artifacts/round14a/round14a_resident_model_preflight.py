#!/usr/bin/env python3
"""No-device fail-closed preflight for a Round14a B request snapshot."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys


ROLES = {306: ("baseline", "A"), 307: ("fplus1", "A"),
         308: ("fplus1", "B"), 309: ("baseline", "B"),
         310: ("baseline", "A"), 311: ("fplus1", "A"),
         312: ("fplus1", "B"), 313: ("baseline", "B"),
         314: ("baseline", "A"), 315: ("fplus1", "A"),
         316: ("fplus1", "B"), 317: ("baseline", "B"),
         318: ("baseline", "A"), 319: ("fplus1", "A"),
         320: ("fplus1", "B"), 321: ("baseline", "B")}
MUTABLE = {"round14a-source-sha256.json", "execution-budget.json",
           "execution-budget.lock", "host-controller.json"}


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tree(root):
    return {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
            and str(p.relative_to(root)) not in MUTABLE
            and not str(p.relative_to(root)).startswith("host-controller")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    devices = sorted(str(path) for path in Path("/dev").glob("davinci[0-9]*"))
    if devices:
        raise RuntimeError(f"offline preflight exposed NPU devices: {devices}")
    manifest = root / "round14a-source-sha256.json"
    if tree(root) != json.loads(manifest.read_text()):
        raise RuntimeError("Round14a resident snapshot manifest mismatch")
    contract = json.loads((root / "ROUND14A_B_RESIDENT_MODEL_CONTRACT.json").read_text())
    expected = ROLES.get(contract.get("version"))
    if expected != (contract.get("strategy"), contract.get("block")):
        raise RuntimeError("Round14a resident role mismatch")
    if (contract.get("generation_limit") != 20 or contract.get("abort_limit") != 0 or
            contract.get("profiler") != "OFF"):
        raise RuntimeError("Round14a resident budget mismatch")
    framework = json.loads((root / "framework-profile.json").read_text())
    streaming = json.loads((root / "streaming-profile.json").read_text())
    if (framework["prefill"]["threshold_tokens"] != 129 or
            streaming != {"schema": 1, "threshold_tokens": 129, "chunk_tokens": 32}):
        raise RuntimeError("Round14a resident arm is not H129/chunk32")
    launcher = (root / "source/tools/launch_hybrid.py").read_text()
    if ("ROUND14_GDN_BACKEND_ENABLE=0" not in launcher or
            "ROUND13_ATTRIBUTION_ENABLE" in launcher):
        raise RuntimeError("Round14a resident arm changed GDN or enabled profiling")
    strategy, block = expected
    if strategy == "fplus1":
        required = ("ROUND14A_RESIDENT_STRATEGY=fplus1", "ROUND14A_RESIDENT_PROFILE=",
                    "ROUND14A_RESIDENT_COMPONENT_GATE=")
        if any(item not in launcher for item in required):
            raise RuntimeError("FPLUS1 launcher environment is incomplete")
        qwen = root / "overlay-python/sglang/srt/models/qwen4_exp.py"
        qwen_source = qwen.read_text()
        if ("ROUND13_PLACEMENT_COMPONENT_GATE" in qwen_source or
                "ROUND14A_RESIDENT_COMPONENT_GATE" not in qwen_source or
                "ROUND14A_RESIDENT_COMPONENT_GATE_SHA256" not in qwen_source):
            raise RuntimeError("FPLUS1 Qwen component-gate environment binding mismatch")
        binding = json.loads((root / "ROUND14A_B_RESIDENT_SOURCE_BINDING.json").read_text())
        if (binding.get("status") != "PASS_ROUND14A_B_RESIDENT_SOURCE_BINDING" or
                binding.get("resident_qwen4_exp_sha256") != sha(qwen)):
            raise RuntimeError("FPLUS1 source binding mismatch")
        qsa_binding = json.loads((root / "ROUND14A_QSA_GDN_SOURCE_BINDING.json").read_text())
        gate_tree = ast.parse((root / "source/runtime/context_reference_gate.py").read_text())
        approved = None
        for node in gate_tree.body:
            if (isinstance(node, ast.Assign) and len(node.targets) == 1 and
                    isinstance(node.targets[0], ast.Name) and
                    node.targets[0].id == "APPROVED_QSA_BINDING"):
                approved = ast.literal_eval(node.value)
        if qsa_binding != approved:
            raise RuntimeError("FPLUS1 QSA/GDN binding JSON differs from the exact gate tuple")
        if "H129 Hybrid inputs <=128" not in (root / "overlay-python/sglang/srt/layers/round9/prefill_binding.py").read_text():
            raise RuntimeError("FPLUS1 fail-closed Layerwise boundary is absent")
    else:
        if "ROUND14A_RESIDENT_STRATEGY=" in launcher:
            raise RuntimeError("baseline arm accidentally enables FPLUS1")
    tools = root / "source/tools"
    sys.path.insert(0, str(tools))
    try:
        from install_candidate_overlay import tree_digest, tree_files
        receipt = json.loads((root / "overlay-python/round9-candidate-overlay.json").read_text())
        if receipt["candidate_tree_sha256"] != tree_digest(tree_files(root / "overlay-python")):
            raise RuntimeError("resident overlay receipt mismatch")
        monitor_source = (tools / "monitor_owned_server.py").read_text()
        if ("kt-round14a-b-resident-(?:baseline|fplus1)-[ab]-v(?:306|307|308|309|310|311|312|313|314|315|316|317|318|319|320|321)" not in
                monitor_source):
            raise RuntimeError("resident owned-container name is not guard-allowlisted")
    finally:
        sys.path.remove(str(tools))
    for name in ("round14a_resident_model_driver.py", "round14a_resident_model_controller.py",
                 "round14a_resident_model_supervisor.py"):
        compile((root / name).read_text(), name, "exec")
    result = {"schema": 1, "status": "PASS_ROUND14A_B_RESIDENT_MODEL_OFFLINE_PREFLIGHT",
              "version": contract["version"], "strategy": strategy, "block": block,
              "physical_devices_exposed": devices, "manifest_sha256": sha(manifest),
              "profiler": "OFF", "threshold": 129, "guard_allowlisted": True}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(result["status"], contract["version"], strategy, block, flush=True)


if __name__ == "__main__": main()
