#!/usr/bin/env python3
"""Run one guarded native-SGLang TP8 BF16 server with original QSA."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time


OLD_QSA_SHA256 = "76e9f34066f0b42f42545427967c0b789d5c32031dc5c085e230efaf95e3c54c"
PROMOTED_QSA_SHA256 = "0fd8d98b37f7e90e90f9866356892236dcc7d0114351d1438025561df6c1bd35"
CONFIG_SHA256 = "889658f2508e8c61d409b02e70e0d78d8d4452ec65aaafbe129805d213d2e74b"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--resolved-profile", type=Path, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    args = parser.parse_args()

    from sglang.srt.server_args import ServerArgs

    values = json.loads(args.profile.read_text())
    server_args = ServerArgs(**values)
    for key, value in values.items():
        if getattr(server_args, key) != value:
            raise RuntimeError(f"SGLang changed frozen native TP8 field {key}")
    if (
        server_args.tp_size != 8 or server_args.device != "npu"
        or server_args.kt_weight_path is not None or server_args.cpu_offload_gb != 0
        or server_args.dtype != "bfloat16" or server_args.attention_backend != "ascend"
        or not server_args.disable_cuda_graph or not server_args.disable_radix_cache
        or not server_args.disable_overlap_schedule or server_args.max_running_requests != 1
        or server_args.chunked_prefill_size != 32 or server_args.max_prefill_tokens != 32
        or server_args.mem_fraction_static != 0.70
    ):
        raise RuntimeError("resolved native TP8 profile violates the frozen contract")
    if args.resolved_profile.exists():
        raise FileExistsError(args.resolved_profile)
    args.resolved_profile.write_text(json.dumps({
        "requested": values, "resolved": {key: getattr(server_args, key) for key in values},
        "profile_sha256": sha256(args.profile),
    }, indent=2) + "\n")
    admission = json.loads(args.admission.read_text())
    if (
        admission.get("status") != "ROUND12A_NATIVE_TP8_READY"
        or admission.get("physical_devices") != list(range(8))
        or admission.get("profile_sha256") != sha256(args.profile)
        or not 0 <= time.time() - admission.get("issued_at", 0) <= 90
        or admission.get("snapshot_manifest_sha256") != os.environ.get("ROUND12A_SNAPSHOT_MANIFEST_SHA256")
        or admission.get("identity_sha256") != os.environ.get("ROUND12A_NATIVE_TP8_IDENTITY_SHA256")
    ):
        raise RuntimeError("missing, stale or unrelated native TP8 admission")
    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    if visible != "0,1,2,3,4,5,6,7":
        raise RuntimeError("native TP8 visible-device mapping changed")
    identity_path = Path(os.environ["ROUND12A_NATIVE_TP8_IDENTITY"])
    identity = json.loads(identity_path.read_text())

    import sglang.srt.layers.round9.ple_binding as ple_binding
    import sglang.srt.layers.round9.qsa as qsa
    import sglang.srt.models.qwen4_exp as qwen4_exp

    observed = {
        "qsa_sha256": sha256(Path(qsa.__file__).resolve()),
        "patched_qwen_sha256": sha256(Path(qwen4_exp.__file__).resolve()),
        "ple_binding_sha256": sha256(Path(ple_binding.__file__).resolve()),
    }
    if (
        observed["qsa_sha256"] != OLD_QSA_SHA256
        or observed["qsa_sha256"] == PROMOTED_QSA_SHA256
        or identity.get("qsa_mode") != "QUALIFIED_ORIGINAL"
        or identity.get("config_sha256") != CONFIG_SHA256
        or any(identity.get(key) != value for key, value in observed.items())
        or identity.get("pure_npu_scope") != "MAIN_AND_EXPERT_WEIGHTS_AND_COMPUTE_ON_EIGHT_NPUS"
        or identity.get("host_dependency") != "PLE_CHECKPOINT_ROW_TABLE_REMAINS_HOST_BACKED"
    ):
        raise RuntimeError("native TP8 imported source differs from frozen identity")
    qwen_source = Path(qwen4_exp.__file__).read_text()
    if "hidden = attn_tp_all_reduce(hidden)" not in qwen_source:
        raise RuntimeError("native TP8 Qwen source lacks the required TP reduction")
    runtime = {
        "schema": 1, "status": "PASS_ROUND12A_NATIVE_TP8_SERVER_PREFLIGHT",
        "tp_size": 8, "physical_devices": list(range(8)), "precision": "BF16",
        "attention_backend": "ascend", "qsa_mode": "QUALIFIED_ORIGINAL",
        "cpu_offload_gb": 0, "kt_weight_path": None, **observed,
        "profile_sha256": sha256(args.profile), "identity_sha256": sha256(identity_path),
        "pure_npu_scope": identity["pure_npu_scope"], "host_dependency": identity["host_dependency"],
        "pci_numa_locality": "UNKNOWN_KERNEL_REPORTS_MINUS_ONE",
        "overlap_policy": "NO_OVERLAP_WITH_PHYSICAL12_PERFORMANCE_WINDOWS",
    }
    Path("/out/native-tp8-runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    print(runtime["status"], runtime, flush=True)
    os.environ["ROUND12A_NATIVE_TP8_MODE"] = "1"

    from sglang.launch_server import run_server
    from sglang.srt.utils import kill_process_tree

    try:
        run_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
