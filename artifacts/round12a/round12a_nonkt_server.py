#!/usr/bin/env python3
"""Run one guarded Round12a full-BF16 non-KT OffloaderV1 server."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import time


ALLOCATOR_LIMIT_MIB = 36352
TRAINING_ID = "0419bcfee46f317d6833e038c28e81cf3d3ecf47323a423062bac2764e0ab2ef"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--resolved-profile", type=Path, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    args = parser.parse_args()

    from sglang.srt.layers.round9.round12a_nonkt_identity import CPU_OFFLOAD_GIB
    from sglang.srt.server_args import ServerArgs

    values = json.loads(args.profile.read_text())
    server_args = ServerArgs(**values)
    for key, value in values.items():
        if getattr(server_args, key) != value:
            raise RuntimeError(f"SGLang changed frozen non-KT field {key}")
    if (
        server_args.kt_weight_path is not None
        or server_args.cpu_offload_gb != CPU_OFFLOAD_GIB
        or server_args.offload_group_size > 0
        or server_args.tp_size != 1
        or server_args.device != "npu"
        or not server_args.disable_cuda_graph
        or not server_args.disable_radix_cache
        or not server_args.disable_overlap_schedule
        or server_args.max_running_requests != 1
        or server_args.chunked_prefill_size != 32
        or server_args.max_prefill_tokens != 32
    ):
        raise RuntimeError("resolved non-KT profile violates the frozen contract")
    resolved = {
        "requested": values,
        "resolved": {key: getattr(server_args, key) for key in values},
        "profile_sha256": sha256(args.profile),
    }
    if args.resolved_profile.exists():
        raise FileExistsError(args.resolved_profile)
    args.resolved_profile.write_text(json.dumps(resolved, indent=2) + "\n")

    admission = json.loads(args.admission.read_text())
    if (
        admission.get("status") != "ROUND12A_NONKT_READY"
        or admission.get("training_container_id") != TRAINING_ID
        or not 0 <= time.time() - admission.get("issued_at", 0) <= 60
        or admission.get("profile_sha256") != resolved["profile_sha256"]
        or admission.get("allocator_limit_mib") != ALLOCATOR_LIMIT_MIB
        or admission.get("owned_limit_mib") != 37376
        or admission.get("reserve_hbm_mib") != 8704
        or admission.get("cpu_offload_gib") != CPU_OFFLOAD_GIB
        or admission.get("snapshot_manifest_sha256")
        != os.environ.get("ROUND12A_SNAPSHOT_MANIFEST_SHA256")
        or admission.get("identity_sha256")
        != os.environ.get("ROUND12A_NONKT_IDENTITY_SHA256")
    ):
        raise RuntimeError("missing, stale or unrelated Round12a non-KT admission")
    if re.fullmatch(r"[0-9a-f]{64}", TRAINING_ID) is None:
        raise RuntimeError("training identity malformed")

    os.environ["ROUND9_RESOURCE_MODE"] = "shared"
    os.environ["ROUND9_ALLOCATOR_LIMIT_MIB"] = str(ALLOCATOR_LIMIT_MIB)
    os.environ["ROUND12A_NONKT_MODE"] = "1"

    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(0)
    free, total = torch.npu.mem_get_info()
    cap_bytes = ALLOCATOR_LIMIT_MIB * 1024**2
    if total <= cap_bytes:
        raise RuntimeError("NPU visible capacity cannot support the non-KT cap")
    torch.npu.set_per_process_memory_fraction(cap_bytes / total)
    acl = ctypes.CDLL(
        "/usr/local/Ascend/cann-9.0.0/aarch64-linux/lib64/libascendcl.so"
    )
    physical = ctypes.c_int32(-1)
    acl.aclrtGetPhyDevIdByLogicDevId.argtypes = [
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
    ]
    if (
        acl.aclrtGetPhyDevIdByLogicDevId(0, ctypes.byref(physical)) != 0
        or physical.value != 12
        or torch.npu.device_count() != 1
        or list(Path("/dev").glob("davinci[0-9]*")) != [Path("/dev/davinci12")]
    ):
        raise RuntimeError("non-KT server does not map only to physical12")

    import sglang.srt.layers.round9.context_reference_binding as context_binding
    import sglang.srt.layers.round9.context_reference_runtime as context_runtime
    import sglang.srt.layers.round9.qsa as qsa
    import sglang.srt.model_executor.model_runner as model_runner
    import sglang.srt.models.qwen4_exp as qwen4_exp
    import sglang.srt.utils.offloader as offloader

    identity_path = Path(os.environ["ROUND12A_NONKT_IDENTITY"])
    identity = json.loads(identity_path.read_text())
    observed = {
        "qwen4_exp_sha256": sha256(Path(qwen4_exp.__file__).resolve()),
        "model_runner_sha256": sha256(Path(model_runner.__file__).resolve()),
        "offloader_sha256": sha256(Path(offloader.__file__).resolve()),
        "qsa_sha256": sha256(Path(qsa.__file__).resolve()),
        "context_reference_runtime_sha256": sha256(
            Path(context_runtime.__file__).resolve()
        ),
        "context_reference_binding_sha256": sha256(
            Path(context_binding.__file__).resolve()
        ),
    }
    if any(identity.get(key) != value for key, value in observed.items()):
        raise RuntimeError("non-KT imported source differs from the frozen identity")
    runtime = {
        "schema": 1,
        "status": "PASS_ROUND12A_NONKT_SERVER_PREFLIGHT",
        "physical_device": 12,
        "cpu_offload_gb": CPU_OFFLOAD_GIB,
        "expert_precision": "BF16",
        "offloader": "SGLANG_OFFLOADERV1_TIED_AWARE",
        **observed,
        "allocator_limit_mib": ALLOCATOR_LIMIT_MIB,
        "numa": {"policy": "INTERLEAVE", "nodes": [2, 3, 4, 7]},
        "profile_sha256": resolved["profile_sha256"],
        "identity_sha256": sha256(identity_path),
        "visible_total_bytes": total,
        "visible_free_before_server_bytes": free,
    }
    Path("/out/nonkt-runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    print("PASS_ROUND12A_NONKT_SERVER_PREFLIGHT", runtime, flush=True)

    from sglang.launch_server import run_server
    from sglang.srt.utils import kill_process_tree

    try:
        run_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
