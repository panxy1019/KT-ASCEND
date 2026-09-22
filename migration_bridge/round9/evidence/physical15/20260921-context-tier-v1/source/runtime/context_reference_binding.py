"""Default-off real-model binding for the bounded QSA reference consumer."""

import hashlib
import json
import os
from pathlib import Path

from .context_reference_runtime import (
    ContextReferenceRuntime, load_context_reference_profile,
)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _serial_profile(args):
    return (
        args.disable_overlap_schedule and args.disable_radix_cache
        and args.disable_cuda_graph and args.max_running_requests == 1
        and args.tp_size == 1 and args.device == "npu"
        and getattr(args, "speculative_algorithm", None) is None
        and args.chunked_prefill_size == 32 and args.max_prefill_tokens == 32
    )


def context_reference_prepare(runner):
    profile = load_context_reference_profile()
    if profile is None:
        return
    if not _serial_profile(runner.server_args):
        raise RuntimeError("context reference requires serial TP1 no-prefix bounded32 scheduler")
    if getattr(runner, "_round9_context_reference_gate", None) is not None:
        raise RuntimeError("context reference preparation may run only once")
    args = runner.server_args
    from .candidate_bank import bind_verified_banks
    binding = bind_verified_banks(
        os.environ["ROUND9_CANDIDATE_PROFILE"], args.kt_weight_path, args.model_path)
    from . import (context_profile, context_page_store, kv_quant_reference,
                   context_staging, qsa)
    from .context_reference_gate import require_context_reference_gate
    import torch
    import torch_npu
    receipt = profile["device_gate"]
    gate = require_context_reference_gate(
        receipt["path"], receipt["sha256"],
        source_identity=binding["source_identity"],
        source_sha256={
            module.__name__.rsplit(".", 1)[-1] + ".py": _sha(module.__file__)
            for module in (context_profile, context_page_store,
                           kv_quant_reference, context_staging, qsa)
        },
        torch_version=str(torch.__version__),
        torch_npu_version=str(torch_npu.__version__),
    )
    from .context_staging import MAX_FINAL_STAGING_BYTES
    runner._round9_context_reference_gate = gate
    runner._round9_context_reference_profile = profile
    runner._round9_context_workspace_bytes = (
        gate["peak_extra_bytes"] + MAX_FINAL_STAGING_BYTES)


def context_reference_ready(runner):
    profile = load_context_reference_profile()
    if profile is None:
        return
    if (getattr(runner, "_round9_context_reference_gate", None) is None
            or getattr(runner, "_round9_context_reference", None) is not None):
        raise RuntimeError("context reference preparation/ready lifecycle is invalid")
    owners = []
    for module in runner.model.modules():
        indexer = getattr(module, "indexer", None)
        if type(indexer).__name__ == "Qwen4ReferenceIndexer":
            owners.append((int(indexer.layer_id), module))
    if [layer for layer, unused in owners] != list(range(3, 48, 4)):
        raise RuntimeError("context reference requires all12 ordered QSA layers")
    log = Path("/out/context-reference-events.jsonl")
    if log.exists():
        raise FileExistsError("context reference event log must be fresh")

    def event(item):
        with log.open("a") as stream:
            stream.write(json.dumps(item, allow_nan=False) + "\n")

    runtime = ContextReferenceRuntime(profile["mode"], event=event)
    for layer, owner in owners:
        if getattr(owner, "_round9_context_consumer", None) is not None:
            raise RuntimeError("context reference consumer already bound")
        owner._round9_context_consumer = runtime
    runner._round9_context_reference = runtime
    runner._round9_context_request = None
    event({
        "status": "CONTEXT_REFERENCE_BOUND",
        "mode": profile["mode"],
        "device_gate_sha256": profile["device_gate"]["sha256"],
        "qsa_layers": len(owners),
        "workspace_bytes": runner._round9_context_workspace_bytes,
        "hbm_release": False,
        "qualification": "COMPONENT_GATE_BOUND_FULL_MODEL_PENDING",
    })


def context_before(scheduler, batch):
    if load_context_reference_profile() is None:
        return
    runner = scheduler.tp_worker.model_runner
    runtime = getattr(runner, "_round9_context_reference", None)
    if runtime is None or len(batch.reqs) != 1:
        raise RuntimeError("enabled context reference requires one bound serial request")
    request = batch.reqs[0]
    previous = runtime.request_token
    if previous is not None and previous is not request:
        reason = getattr(previous, "finished_reason", None)
        if (runtime.active or reason is None
                or reason.__class__.__name__ != "FINISH_ABORT"):
            raise RuntimeError("context request switched without a drained abort")
        runtime.abort(previous)
    runtime.begin(request.rid, request)
    runner._round9_context_request = request


def context_after(scheduler):
    if load_context_reference_profile() is None:
        return
    runner = scheduler.tp_worker.model_runner
    runtime = getattr(runner, "_round9_context_reference", None)
    request = getattr(runner, "_round9_context_request", None)
    if runtime is None or request is None:
        raise RuntimeError("context reference batch is unleased")
    runtime.after(request, finished=bool(request.finished()))
    runner._round9_context_request = None
