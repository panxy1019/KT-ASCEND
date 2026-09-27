#!/usr/bin/env python3
"""Import the frozen TP8 tuple in the runtime image without exposing an NPU."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sys


OLD_QSA_SHA256 = "76e9f34066f0b42f42545427967c0b789d5c32031dc5c085e230efaf95e3c54c"
PROMOTED_QSA_SHA256 = "0fd8d98b37f7e90e90f9866356892236dcc7d0114351d1438025561df6c1bd35"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@contextmanager
def no_device_npu_import_probe(torch_module):
    """Let SGLang select its NPU import path without exposing a device.

    The frozen SGLang ``is_npu()`` helper raises during import when torch_npu
    is installed but no device node is visible.  The same-image preflight
    intentionally has no device nodes, so scope a process-local availability
    probe override to imports and ServerArgs construction, then restore it.
    """

    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES"):
        raise RuntimeError("same-image source preflight must not expose an NPU")
    npu = getattr(torch_module, "npu", None)
    original = getattr(npu, "is_available", None)
    if not callable(original):
        raise RuntimeError("torch_npu did not install the NPU availability probe")
    if original():
        raise RuntimeError("an NPU is unexpectedly available in no-device preflight")
    receipt = {
        "availability_before": False,
        "availability_for_import": True,
        "restored": False,
        "scope": "IMPORT_AND_SERVERARGS_ONLY",
    }
    npu.is_available = lambda: True
    try:
        yield receipt
    finally:
        npu.is_available = original
        receipt["restored"] = True


def run(snapshot: Path) -> dict:
    snapshot = Path(snapshot).resolve(strict=True)
    overlay = snapshot / "overlay-python"
    profile_path = snapshot / "native-tp8-server-profile.json"
    identity_path = snapshot / "ROUND12A_NATIVE_TP8_IDENTITY.json"
    profile = json.loads(profile_path.read_text())
    identity = json.loads(identity_path.read_text())
    sys.path.insert(0, str(overlay))
    try:
        import torch
        import torch_npu
        with no_device_npu_import_probe(torch) as import_probe:
            from sglang.srt.server_args import ServerArgs
            import sglang.srt.layers.round9.ple_binding as ple_binding
            import sglang.srt.layers.round9.qsa as qsa
            import sglang.srt.mem_cache.round9_qsa_pool as qsa_pool
            import sglang.srt.models.qwen4_exp as qwen4_exp

            args = ServerArgs(**profile)
            resolved = {key: getattr(args, key) for key in profile}
            if resolved != profile:
                raise RuntimeError("runtime image normalized a frozen native TP8 server field")
            observed = {
                "qsa_sha256": sha256(Path(qsa.__file__).resolve()),
                "patched_qwen_sha256": sha256(Path(qwen4_exp.__file__).resolve()),
                "ple_binding_sha256": sha256(Path(ple_binding.__file__).resolve()),
            }
            if (
                observed["qsa_sha256"] != OLD_QSA_SHA256
                or observed["qsa_sha256"] == PROMOTED_QSA_SHA256
                or any(identity.get(key) != value for key, value in observed.items())
                or args.tp_size != 8
                or args.device != "npu"
                or args.kt_weight_path is not None
                or args.cpu_offload_gb != 0
                or not hasattr(qsa_pool, "Round9QSAKVPool")
            ):
                raise RuntimeError("runtime image imported a different native TP8 tuple")
        return {
            "schema": 1,
            "status": "PASS_ROUND12A_NATIVE_TP8_SAME_IMAGE_PREFLIGHT",
            "snapshot": str(snapshot),
            "snapshot_manifest_sha256": sha256(snapshot / "round12-source-sha256.json"),
            "profile_sha256": sha256(profile_path),
            "identity_sha256": sha256(identity_path),
            "torch_version": str(torch.__version__),
            "torch_npu_version": str(torch_npu.__version__),
            "tp_size": args.tp_size,
            "physical_devices_exposed": [],
            "no_device_import_probe": import_probe,
            **observed,
        }
    finally:
        sys.path.remove(str(overlay))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = run(args.snapshot)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(report["status"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
