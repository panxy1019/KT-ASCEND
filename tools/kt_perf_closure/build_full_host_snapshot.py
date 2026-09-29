"""Derive an all-layer Host-parent diagnostic snapshot from frozen v506."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise ValueError(f"{label}: expected one anchor, found {count}")
    return source.replace(old, new)


def patch_observer(source: str) -> str:
    source = replace_once(
        source,
        "GDN_LAYERS = (0, 24, 46)\n"
        "QSA_LAYERS = (3, 23, 47)\n"
        "EXPERT_LAYERS = (0, 24, 47)",
        "GDN_LAYERS = tuple(layer for layer in range(48) if layer % 4 != 3)\n"
        "QSA_LAYERS = tuple(range(3, 48, 4))\n"
        "GDN_DEVICE_LAYERS = (0, 24, 46)\n"
        "QSA_DEVICE_LAYERS = (3, 23, 47)\n"
        "EXPERT_LAYERS = (0, 24, 47)",
        "all-layer observer constants",
    )
    source = replace_once(
        source,
        'wrap(module, "forward", "GDN", layer, device=True)',
        'wrap(module, "forward", "GDN", layer, device=layer in GDN_DEVICE_LAYERS)',
        "GDN device sample",
    )
    source = replace_once(
        source,
        'wrap(module, "forward", "QSA_INDEXER", layer, device=True)',
        'wrap(module, "forward", "QSA_INDEXER", layer, device=layer in QSA_DEVICE_LAYERS)',
        "QSA indexer device sample",
    )
    return replace_once(
        source,
        'wrap(module, "self_attention", "QSA_ATTENTION", layer, device=True)',
        'wrap(module, "self_attention", "QSA_ATTENTION", layer, device=layer in QSA_DEVICE_LAYERS)',
        "QSA attention device sample",
    )


def patch_launcher(
    source: str,
    *,
    run_version: int,
    baseline_observer_sha: str,
    diagnostic_observer_sha: str,
) -> str:
    if any(len(value) != 64 for value in (baseline_observer_sha, diagnostic_observer_sha)):
        raise ValueError("observer SHA-256 is malformed")
    source = replace_once(
        source,
        "if a.version != 506 or a.cost_mode != 'OFF':",
        f"if a.version != {run_version} or a.cost_mode != 'OFF':",
        "version gate",
    )
    source = replace_once(
        source,
        "name='kt-root-cause-profile-a-v506';out=root/'evidence'/'20260929-kt-root-cause-profile-a-v506'",
        f"name='kt-root-cause-full-host-a-v{run_version}';out=root/'evidence'/'20260929-kt-root-cause-full-host-a-v{run_version}'",
        "run identity",
    )
    source = replace_once(
        source,
        "KT_ROOT_TRACE_BINDING_PATH=/round9/kt-root-cause-profile-a-v506/TRACE_NATIVE_BINDING.json",
        f"KT_ROOT_TRACE_BINDING_PATH=/round9/kt-root-cause-full-host-a-v{run_version}/TRACE_NATIVE_BINDING.json",
        "trace binding path",
    )
    old = (
        "candidate_files[_layerwise_relative]='b4605b8f4ab0fb2a56c2f8b8eea693c440d1d15af1b1617184d60eb6be7684a9'\n"
        "if overlay['baseline_tree_sha256']!=tree_digest(tree_files(root/'src/sglang-round9-port/python')) or overlay['candidate_tree_sha256']!=tree_digest(candidate_files):"
    )
    new = (
        "candidate_files[_layerwise_relative]='b4605b8f4ab0fb2a56c2f8b8eea693c440d1d15af1b1617184d60eb6be7684a9'\n"
        "_observer_relative='sglang/srt/layers/round9/round13_attribution.py'\n"
        f"if candidate_files.get(_observer_relative)!='{diagnostic_observer_sha}':\n"
        " raise RuntimeError('full-host observer differs from bound source')\n"
        f"candidate_files[_observer_relative]='{baseline_observer_sha}'\n"
        "if overlay['baseline_tree_sha256']!=tree_digest(tree_files(root/'src/sglang-round9-port/python')) or overlay['candidate_tree_sha256']!=tree_digest(candidate_files):"
    )
    return replace_once(source, old, new, "observer overlay normalization")


def patch_monitor(source: str, *, run_version: int) -> str:
    return replace_once(
        source,
        "r'|kt-round16-a1-[ab]-v003|kt-root-cause-profile-[ab]-v506)', name) is not None",
        f"r'|kt-round16-a1-[ab]-v003|kt-root-cause-profile-[ab]-v506|kt-root-cause-full-host-a-v{run_version})', name) is not None",
        "monitor allow-list",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--donor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--run-version", type=int, default=508)
    args = parser.parse_args()
    expected = f"kt-root-cause-full-host-a-v{args.run_version}"
    if args.output.name != expected:
        raise ValueError(f"output name must be {expected}")
    if args.output.exists() or args.receipt.exists():
        raise FileExistsError("snapshot and receipt must be fresh")
    shutil.copytree(args.donor, args.output, symlinks=True)
    observer = args.output / "overlay-python/sglang/srt/layers/round9/round13_attribution.py"
    launcher = args.output / "source/tools/launch_hybrid.py"
    monitor = args.output / "source/tools/monitor_owned_server.py"
    baseline_observer_sha = sha256(observer)
    observer.write_text(patch_observer(observer.read_text()))
    diagnostic_observer_sha = sha256(observer)
    launcher.write_text(
        patch_launcher(
            launcher.read_text(),
            run_version=args.run_version,
            baseline_observer_sha=baseline_observer_sha,
            diagnostic_observer_sha=diagnostic_observer_sha,
        )
    )
    monitor.write_text(patch_monitor(monitor.read_text(), run_version=args.run_version))
    for path in (observer, launcher, monitor):
        compile(path.read_text(), str(path), "exec")
    budget_path = args.output / "execution-budget.json"
    budget = json.loads(budget_path.read_text())
    budget["model_starts_used"] = 0
    budget["starts"] = []
    budget_path.write_text(json.dumps(budget, indent=2) + "\n")
    for relative in ("execution-budget.lock", "host-controller.json"):
        path = args.output / relative
        if path.exists():
            path.unlink()
    receipt = {
        "schema": 1,
        "status": "PASS_KT_ROOT_FULL_HOST_SNAPSHOT",
        "created_at": time.time(),
        "donor": str(args.donor),
        "output": str(args.output),
        "run_version": args.run_version,
        "baseline_observer_sha256": baseline_observer_sha,
        "diagnostic_observer_sha256": diagnostic_observer_sha,
        "launcher_sha256": sha256(launcher),
        "monitor_sha256": sha256(monitor),
        "trace_binding_sha256": sha256(args.output / "TRACE_NATIVE_BINDING.json"),
        "scope": "ALL_48_ATTENTION_PARENTS_HOST_CLOCK_DEVICE_EVENTS_REMAIN_SAMPLED",
        "math_changes": 0,
        "synchronization_changes": 0,
    }
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(receipt["status"])


if __name__ == "__main__":
    main()
