#!/usr/bin/env python3
"""Build one immutable Round13 B QSA Decode model arm from v177 S0."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import stat
import sys

from build_round13_b import bind_candidate_runtime, patch_context_runtime


ROLES = {
    217: ("baseline", "smoke", "A", "smoke-a"),
    218: ("candidate", "smoke", "B", "smoke-b"),
    219: ("baseline", "request", "A", "baseline-a"),
    220: ("candidate", "request", "A", "candidate-a"),
    221: ("candidate", "request", "B", "candidate-b"),
    222: ("baseline", "request", "B", "baseline-b"),
}
MUTABLE = {
    "round13-source-sha256.json",
    "execution-budget.json",
    "execution-budget.lock",
    "host-controller.json",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replace_once(source, old, new, label):
    if source.count(old) != 1:
        raise ValueError(f"{label} precondition mismatch")
    return source.replace(old, new)


def patch_launcher(source, *, version, label):
    source = replace_once(
        source,
        "if a.version != 177 or a.cost_mode != 'OFF':",
        f"if a.version != {version} or a.cost_mode != 'OFF':",
        "Round13 B launcher version",
    )
    source = replace_once(
        source,
        "name='kt-round12a-kt-b-v177';out=root/'evidence'/'20260927-round12a-kt-b-v177'",
        f"name='kt-round13-b-{label}-v{version}';out=root/'evidence'/'20260927-round13-b-{label}-v{version}'",
        "Round13 B launcher identity",
    )
    source = replace_once(
        source,
        "if budget.get('stage')!='ROUND12A_ABBA':raise RuntimeError('wrong Round12a ABBA budget')",
        "if budget.get('stage')!='ROUND13_B_QSA_DECODE':raise RuntimeError('wrong Round13 B budget')",
        "Round13 B launcher budget",
    )
    compile(source, "launch_hybrid.py", "exec")
    return source


def refresh_overlay_receipt(root, *, implementation, runtime_sha256):
    overlay = root / "overlay-python"
    sys.path.insert(0, str(root / "source/tools"))
    try:
        from install_candidate_overlay import tree_digest, tree_files

        receipt_path = overlay / "round9-candidate-overlay.json"
        receipt = json.loads(receipt_path.read_text())
        receipt["candidate_files"] = tree_files(overlay)
        receipt["candidate_tree_sha256"] = tree_digest(
            receipt["candidate_files"]
        )
        receipt["round13_b"] = {
            "status": "INSTALLED_DIRECT_RESIDENT_BF16_ROW_DECODE"
            if implementation == "candidate"
            else "UNCHANGED_ROUND12A_PROMOTED_QSA",
            "implementation": implementation,
            "context_reference_runtime_sha256": runtime_sha256,
        }
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    finally:
        sys.path.remove(str(root / "source/tools"))


def source_manifest(root):
    return {
        str(path.relative_to(root)): sha(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and "__pycache__" not in path.parts
        and str(path.relative_to(root)) not in MUTABLE
        and not str(path.relative_to(root)).startswith("host-controller")
    }


def build(args):
    implementation, mode, block, label = ROLES[args.version]
    if args.output.exists():
        raise FileExistsError(args.output)
    shutil.copytree(
        args.parent,
        args.output,
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    for path in (args.output, *args.output.rglob("*")):
        path.chmod(path.stat().st_mode | stat.S_IWUSR)
    for name in (
        "host-controller.json",
        "execution-budget.lock",
        "round12-source-sha256.json",
        "ROUND12A_MODEL_CONTRACT.json",
        "ROUND12A_ABBA_QUEUE.json",
    ):
        path = args.output / name
        if path.exists():
            path.unlink()
    input_copies = {
        "round13_b_model_driver.py": "round13_b_model_driver.py",
        "round13_model_controller.py": "round13_model_controller.py",
        "round13_model_supervisor.py": "round13_model_supervisor.py",
        "round13_monitor_owned.py": "source/tools/monitor_owned_server.py",
    }
    parent_copies = {
        "round11_t4_driver.py": "round11_t4_driver.py",
        "round10_t1_driver.py": "round10_t1_driver.py",
        "round10_t1_protocol.py": "round10_t1_protocol.py",
        "round10-t1-fixture.json": "round10-t1-fixture.json",
        "round11-t1-fixture.json": "round11-t1-fixture.json",
    }
    for source, target in input_copies.items():
        shutil.copy2(args.inputs / source, args.output / target)
    for source, target in parent_copies.items():
        shutil.copy2(args.parent / source, args.output / target)

    host_runtime = args.output / "source/runtime/context_reference_runtime.py"
    overlay_runtime = (
        args.output
        / "overlay-python/sglang/srt/layers/round9/context_reference_runtime.py"
    )
    baseline_runtime_sha = sha(host_runtime)
    if baseline_runtime_sha != sha(overlay_runtime):
        raise ValueError("Round13 B parent host/overlay runtime mismatch")
    delta = None
    if implementation == "candidate":
        host_runtime.write_text(patch_context_runtime(host_runtime.read_text()))
        overlay_runtime.write_text(patch_context_runtime(overlay_runtime.read_text()))
        if sha(host_runtime) != sha(overlay_runtime):
            raise RuntimeError("Round13 B candidate runtime copies differ")
        delta = bind_candidate_runtime(args.output, sha(host_runtime))
    runtime_sha = sha(host_runtime)
    refresh_overlay_receipt(
        args.output,
        implementation=implementation,
        runtime_sha256=runtime_sha,
    )

    launcher = args.output / "source/tools/launch_hybrid.py"
    launcher.write_text(
        patch_launcher(
            launcher.read_text(), version=args.version, label=label
        )
    )
    for name in ("framework-profile.json", "streaming-profile.json"):
        path = args.output / name
        document = json.loads(path.read_text())
        if name.startswith("framework"):
            if document["prefill"]["threshold_tokens"] != 129:
                raise ValueError("Round13 B parent framework threshold mismatch")
            document["prefill"]["threshold_tokens"] = 513
        else:
            if document["threshold_tokens"] != 129:
                raise ValueError("Round13 B parent streaming threshold mismatch")
            document["threshold_tokens"] = 513
        path.write_text(json.dumps(document, indent=2) + "\n")

    generation_limit = 8 if mode == "smoke" else 20
    abort_limit = 1 if mode == "smoke" else 0
    budget = {
        "schema": 1,
        "stage": "ROUND13_B_QSA_DECODE",
        "version": args.version,
        "implementation": implementation,
        "mode": mode,
        "block": block,
        "model_starts_limit": 1,
        "model_starts_used": 0,
        "starts": [],
        "generation_limit": generation_limit,
        "abort_limit": abort_limit,
        "automatic_retries_limit": 0,
    }
    (args.output / "execution-budget.json").write_text(
        json.dumps(budget, indent=2) + "\n"
    )
    driver_pass = (
        "PASS_ROUND13_B_SMOKE_RAW"
        if mode == "smoke"
        else "PASS_ROUND13_B_REQUEST_RAW"
    )
    process_pass = (
        "PASS_ROUND13_B_SMOKE_PROCESS"
        if mode == "smoke"
        else "PASS_ROUND13_B_REQUEST_PROCESS"
    )
    contract = {
        "schema": 1,
        "status": "FROZEN_NOT_RUN",
        "stage": "ROUND13_B_QSA_DECODE",
        "version": args.version,
        "implementation": implementation,
        "mode": mode,
        "block": block,
        "container": f"kt-round13-b-{label}-v{args.version}",
        "evidence": f"20260927-round13-b-{label}-v{args.version}",
        "generation_limit": generation_limit,
        "abort_limit": abort_limit,
        "profiler": "OFF",
        "profile_threshold": 513,
        "default_threshold": 129,
        "baseline_runtime_sha256": baseline_runtime_sha,
        "runtime_sha256": runtime_sha,
        "runtime_delta_sha256": (
            sha(args.output / "ROUND13_B_QSA_SOURCE_DELTA.json")
            if delta is not None
            else None
        ),
        "raw_result": "round13-b-raw.json",
        "final_result": "round13-b-final.json",
        "driver_pass_status": driver_pass,
        "process_pass_status": process_pass,
        "driver_command": [
            "{DRIVER_ROOT}/round13_b_model_driver.py",
            "--version",
            str(args.version),
            "--implementation",
            implementation,
            "--mode",
            mode,
            "--block",
            block,
            "--out",
            "/out",
            "--fixture10",
            "{DRIVER_ROOT}/round10-t1-fixture.json",
            "--fixture11",
            "{DRIVER_ROOT}/round11-t1-fixture.json",
        ],
    }
    (args.output / "ROUND13_MODEL_CONTRACT.json").write_text(
        json.dumps(contract, indent=2) + "\n"
    )
    (args.output / "round13-source-sha256.json").write_text(
        json.dumps(source_manifest(args.output), indent=2, sort_keys=True) + "\n"
    )
    print(
        "PASS_ROUND13_B_SNAPSHOT",
        args.version,
        implementation,
        mode,
        block,
        len(source_manifest(args.output)),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--version", type=int, choices=tuple(ROLES), required=True)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
