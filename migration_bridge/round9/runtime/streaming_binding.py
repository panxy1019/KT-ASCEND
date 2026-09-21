"""Default-off scheduler binding for the three-plane streaming coordinator."""

import copy
import hashlib
import json
import os
from pathlib import Path

from .framework_config import load_framework
from .cancel_lifecycle import (
    scheduler_before_normal_submit,
    scheduler_request_admitted,
    scheduler_request_retired,
)
from .streaming_prefill import StreamingPrefillCoordinator


def parse_streaming_profile(document):
    if (not isinstance(document, dict)
            or set(document) != {"schema", "threshold_tokens", "chunk_tokens"}
            or type(document["schema"]) is not int or document["schema"] != 1
            or type(document["threshold_tokens"]) is not int
            or document["threshold_tokens"] < 1
            or type(document["chunk_tokens"]) is not int
            or document["chunk_tokens"] != 32):
        raise ValueError("invalid bounded streaming profile")
    return copy.deepcopy(document)


def load_streaming_profile():
    path = os.environ.get("ROUND9_STREAMING_PROFILE")
    expected = os.environ.get("ROUND9_STREAMING_SHA256")
    if not path and not expected:
        return None
    if not path or not expected or not Path(path).is_absolute():
        raise ValueError("streaming profile and SHA are required together")
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("streaming profile changed")
    return parse_streaming_profile(json.loads(raw))


def validate_streaming_stack(streaming, framework, context_reference):
    streaming = parse_streaming_profile(streaming)
    if (framework is None or framework.get("prefill") is None
            or framework.get("ple") is None or context_reference is None
            or framework["prefill"]["threshold_tokens"]
                != streaming["threshold_tokens"]):
        raise ValueError(
            "three-plane streaming requires matching prefill, async PLE and context reference")
    return streaming


def streaming_model_ready(runner):
    profile = load_streaming_profile()
    if profile is None:
        return
    framework = load_framework()
    context = getattr(runner, "_round9_context_reference_profile", None)
    try:
        validate_streaming_stack(profile, framework, context)
    except ValueError as error:
        raise RuntimeError(str(error)) from error
    if any(getattr(runner, name, None) is None for name in (
            "_round9_prefill", "_round9_ple", "_round9_context_reference")):
        raise RuntimeError("streaming requires all three bound planes and matching threshold")
    if getattr(runner, "_round9_streaming", None) is not None:
        raise RuntimeError("streaming coordinator already bound")
    log = Path("/out/streaming-events.jsonl")
    if log.exists():
        raise FileExistsError("streaming event log must be fresh")

    def event(item):
        with log.open("a") as stream:
            stream.write(json.dumps(item, allow_nan=False) + "\n")

    runner._round9_streaming = StreamingPrefillCoordinator(
        profile["threshold_tokens"], chunk_tokens=profile["chunk_tokens"],
        event=event)
    runner._round9_streaming_request = None
    runner._round9_streaming_ple_before = 0
    event({
        "status": "STREAMING_BOUND", "threshold_tokens": profile["threshold_tokens"],
        "chunk_tokens": profile["chunk_tokens"],
        "planes": ["expert", "ple", "context"],
        "qualification": "LOCAL_LIFECYCLE_FULL_MODEL_PENDING",
    })


def streaming_submit_fence(scheduler, batch):
    """Reject a cancelled EXTEND before any Round9 plane begins its batch."""
    if load_streaming_profile() is None:
        return
    if len(batch.reqs) != 1:
        raise RuntimeError("enabled streaming mode requires one serial request")
    if not batch.forward_mode.is_extend():
        return
    request = batch.reqs[0]
    # The context allocation epoch is created later in framework_before.  The
    # lifecycle deliberately permits filling it in on the same live identity.
    scheduler_request_admitted(scheduler, request, None)
    if not scheduler_before_normal_submit(
            scheduler, request, "STREAMING_CHUNK"):
        raise RuntimeError(
            "post-cancel streaming chunk reached model-submit boundary")


def streaming_before(scheduler, batch):
    if load_streaming_profile() is None:
        return
    runner = scheduler.tp_worker.model_runner
    coordinator = getattr(runner, "_round9_streaming", None)
    if coordinator is None or len(batch.reqs) != 1:
        raise RuntimeError("enabled streaming mode requires one bound serial request")
    request = batch.reqs[0]
    previous = coordinator.request_token
    if previous is not None and previous is not request:
        reason = getattr(previous, "finished_reason", None)
        if (coordinator.active or reason is None
                or reason.__class__.__name__ != "FINISH_ABORT"):
            raise RuntimeError("streaming request switched without drained abort")
        coordinator.abort(previous)
    runner._round9_streaming_request = request
    if not batch.forward_mode.is_extend():
        return
    context = runner._round9_context_reference
    if context.identity is None:
        raise RuntimeError("streaming context allocation identity is missing")
    scheduler_request_admitted(
        scheduler, request, context.identity.allocation_epoch)
    boundary = getattr(scheduler, "_round9_dynamic_boundary", None)
    placement_epoch = boundary.generation if boundary is not None else 0
    started = coordinator.begin(
        request.rid, request, placement_epoch=placement_epoch,
        allocation_epoch=context.identity.allocation_epoch,
        total=len(request.origin_input_ids), start=request.last_extend_prefix_len,
        count=request.extend_input_len)
    if started:
        if (not runner._round9_prefill.active
                or runner._round9_ple.context is None or not context.active):
            raise RuntimeError("streaming planes did not begin the same scheduler batch")
        runner._round9_streaming_ple_before = runner._round9_ple.async_batches
    return started


def streaming_after(scheduler):
    if load_streaming_profile() is None:
        return
    runner = scheduler.tp_worker.model_runner
    coordinator = getattr(runner, "_round9_streaming", None)
    request = getattr(runner, "_round9_streaming_request", None)
    if coordinator is None:
        raise RuntimeError("streaming scheduler batch is unleased")
    if request is None:
        if coordinator.active:
            raise RuntimeError("streaming scheduler batch is unleased")
        return
    if coordinator.active:
        prefill = runner._round9_prefill
        ple = runner._round9_ple
        context = runner._round9_context_reference
        if prefill.active or ple.context is not None or context.active or context.next_layer != 12:
            raise RuntimeError("streaming planes did not drain before the shared barrier")
        coordinator.mark_expert(layers=48)
        coordinator.mark_ple(
            async_engaged=ple.async_batches > runner._round9_streaming_ple_before,
            drained=True)
        coordinator.mark_context(layers=12, hbm_release=False)
        coordinator.finish_chunk()
    if request.finished():
        termination = None
        if coordinator.complete:
            coordinator.request_end(request)
            termination = "complete"
        elif coordinator.request_token is request:
            coordinator.abort(request)
            termination = "abort"
        if termination is not None:
            from .working_set_timeline import flush_request
            flush_request(termination)
        # Every traced request is admitted on its first bounded EXTEND batch.
        # Release that strong request-object identity after normal completion,
        # trace-only abort completion, or a below-threshold fallback finish.
        scheduler_request_retired(scheduler, request)
    runner._round9_streaming_request = None
