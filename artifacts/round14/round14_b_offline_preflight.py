#!/usr/bin/env python3
"""No-device same-image source/entry preflight for a Round14 B snapshot."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


MUTABLE = {
    "round14-source-sha256.json",
    "execution-budget.json",
    "execution-budget.lock",
    "host-controller.json",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tree(root):
    root = Path(root)
    return {
        str(path.relative_to(root)): sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
        and str(path.relative_to(root)) not in MUTABLE
        and not str(path.relative_to(root)).startswith("host-controller")
    }


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def preflight(root, *, device_paths=None):
    root = Path(root).resolve(strict=True)
    devices = (
        sorted(str(path) for path in Path("/dev").glob("davinci[0-9]*"))
        if device_paths is None else sorted(device_paths)
    )
    if devices:
        raise RuntimeError(f"no-device preflight has exposed device paths: {devices}")
    expected = json.loads((root / "round14-source-sha256.json").read_text())
    if tree(root) != expected:
        raise RuntimeError("Round14 B snapshot manifest mismatch")

    installer = _load(
        "round14_b_install_candidate_overlay_preflight",
        root / "source/tools/install_candidate_overlay.py",
    )
    overlay = root / "overlay-python"
    overlay_receipt = json.loads(
        (overlay / "round9-candidate-overlay.json").read_text()
    )
    overlay_digest = installer.tree_digest(installer.tree_files(overlay))
    if (
        overlay_receipt.get("candidate_tree_sha256") != overlay_digest
        or overlay_receipt.get("round14_b", {}).get("status")
        != "INSTALLED_PRE_DISPATCH_H_BOUNDED_GATE"
    ):
        raise RuntimeError("Round14 B overlay launcher gate mismatch")

    contract = json.loads((root / "ROUND14_B_CONTRACT.json").read_text())
    profile_path = root / "ROUND14_H_BOUNDED_PROFILE.json"
    profile = json.loads(profile_path.read_text())
    policy = _load(
        "round14_bounded_policy_preflight",
        root / "overlay-python/sglang/srt/round14_bounded_policy.py",
    )
    parsed = policy.parse_profile(profile)
    framework = json.loads((root / "framework-profile.json").read_text())
    streaming = json.loads((root / "streaming-profile.json").read_text())
    threshold = parsed["hybrid_threshold_tokens"]
    if (
        contract.get("limits") != {
            key: parsed[key] for key in (
                "max_input_tokens", "max_output_tokens", "max_total_tokens",
                "hybrid_threshold_tokens",
            )
        }
        or framework.get("prefill", {}).get("threshold_tokens") != threshold
        or streaming.get("threshold_tokens") != threshold
    ):
        raise RuntimeError("Round14 B profile/threshold coupling mismatch")

    manager_path = root / "overlay-python/sglang/srt/managers/tokenizer_manager.py"
    manager = manager_path.read_text()
    if (
        manager.count("from sglang.srt.round14_bounded_policy import") != 1
        or manager.count("enforce_from_environment(") != 2
    ):
        raise RuntimeError("Round14 B tokenizer pre-dispatch gate mismatch")
    compile(manager, str(manager_path), "exec")
    launcher_path = root / "source/tools/launch_hybrid.py"
    launcher = launcher_path.read_text()
    required = (
        f"if a.version != {contract['version']}",
        contract["container"], contract["evidence"],
        "ROUND14_H_BOUNDED_PROFILE=/out/round14-h-bounded-profile.json",
        "ROUND14_H_BOUNDED_PROFILE_SHA256=",
        "ROUND14_H_BOUNDED_EVENTS=/out/round14-boundary-events.jsonl",
    )
    if any(value not in launcher for value in required):
        raise RuntimeError("Round14 B launcher identity/gate mismatch")
    compile(launcher, str(launcher_path), "exec")

    driver = _load("round14_b_driver_preflight", root / "round14_b_driver.py")
    fixture = json.loads((root / "ROUND14_B_REQUESTS.json").read_text())
    edge = json.loads((root / "round11-t1-fixture.json").read_text())
    schedule = driver.build_schedule(
        fixture, edge, version=contract["version"],
        profile=contract["profile"], block=contract["block"],
    )
    if len(schedule) != contract["generation_limit"] or contract["abort_limit"] != 1:
        raise RuntimeError("Round14 B schedule/budget mismatch")
    return {
        "schema": 1,
        "status": "PASS_ROUND14_B_OFFLINE_PREFLIGHT",
        "version": contract["version"],
        "profile": contract["profile"],
        "block": contract["block"],
        "physical_devices_exposed": devices,
        "manifest_sha256": sha256(root / "round14-source-sha256.json"),
        "bounded_profile_sha256": sha256(profile_path),
        "request_fixture_sha256": sha256(root / "ROUND14_B_REQUESTS.json"),
        "schedule_requests": len(schedule),
        "abort_controls": contract["abort_limit"],
        "hybrid_threshold_tokens": threshold,
        "tokenizer_gate_calls": manager.count("enforce_from_environment("),
        "launcher_environment_gate": True,
        "overlay_tree_sha256": overlay_digest,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = preflight(args.root)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(report["status"], report["version"], report["profile"], report["block"])


if __name__ == "__main__":
    main()
