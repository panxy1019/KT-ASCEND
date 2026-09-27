"""Default-off Round13 S0 observer and bounded DEV route counter.

This module observes the promoted-QSA S0 runtime without inserting a device
synchronization.  NPU durations are recorded with events and queried only at
later scheduler boundaries.  Route IDs are copied to Host only for the two
``r13-dev-*`` requests; those intrusive calibration requests are excluded from
all timing conclusions.
"""

from __future__ import annotations

import functools
import json
import math
import os
from pathlib import Path
import re
import threading
import time


GDN_LAYERS = (0, 24, 46)
QSA_LAYERS = (3, 23, 47)
EXPERT_LAYERS = (0, 24, 47)
MAX_SPANS_PER_BATCH = 96
DEV_PREFIX = "r13-dev-"
EXPECTED_BINDINGS = {
    "GDN": list(GDN_LAYERS),
    "QSA_ATTENTION": list(QSA_LAYERS),
    "QSA_INDEXER": list(QSA_LAYERS),
    "PLE": [1],
    "EXPERT": list(EXPERT_LAYERS),
}


def validate_scope(values):
    expected = {
        "storage": "MXFP4",
        "placement": "static",
        "capacity": 32,
        "workers": 128,
        "device": "npu",
        "revision": "de4b8e4d43b917e7706784d8bb445c9af86a3540",
    }
    if values != expected:
        raise ValueError("Round13 P0 observer outside frozen S0 scope")


def observe(original, sink, component, layer, context, device_factory=None,
            on_error=None, clock=time.monotonic_ns):
    """Observe one call while preserving its result and exception semantics."""
    @functools.wraps(original)
    def call(*args, **kwargs):
        ctx = context()
        if ctx is None:
            return original(*args, **kwargs)
        errors = []

        def note(error):
            value = type(error).__name__ + ": " + str(error)[:512]
            errors.append(value)
            if on_error is not None:
                try:
                    on_error(value)
                except Exception:
                    pass

        pair = None
        if device_factory is not None:
            try:
                pair = device_factory()
                pair[0].record()
            except Exception as error:
                note(error)
                pair = None
        started = int(clock())
        outcome = "error"
        try:
            value = original(*args, **kwargs)
            outcome = "return"
            return value
        finally:
            ended = int(clock())
            if pair is not None:
                try:
                    pair[1].record()
                except Exception as error:
                    note(error)
                    pair = None
            row = {
                "component": component,
                "layer": layer,
                "start_ns": started,
                "end_ns": ended,
                "duration_ns": ended - started,
                "outcome": outcome,
                "device_ms": None,
                "device_status": (
                    "PENDING" if pair is not None else
                    "NOT_APPLICABLE" if device_factory is None else
                    "OBSERVER_ERROR"
                ),
                "observer_errors": errors,
                **ctx,
            }
            if pair is not None:
                row["_device"] = pair
            try:
                sink(row)
            except Exception as error:
                note(error)
    return call


def resolve_device_spans(batch):
    complete = True
    for row in batch.get("spans", []):
        pair = row.get("_device")
        if pair is None:
            continue
        try:
            if not pair[1].query():
                row["device_status"] = "PENDING"
                complete = False
                continue
            value = float(pair[0].elapsed_time(pair[1]))
            if not math.isfinite(value) or value < 0:
                raise ValueError("nonfinite or negative device duration")
            row["device_ms"] = value
            row["device_status"] = "COMPLETE"
            del row["_device"]
        except Exception as error:
            value = type(error).__name__ + ": " + str(error)[:512]
            row.setdefault("observer_errors", []).append(value)
            batch.setdefault("observer_errors", []).append(value)
            row["device_status"] = "OBSERVER_ERROR"
            row.pop("_device", None)
    return complete


def _atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _append_json(path, value):
    with path.open("a") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _flush_ready(state):
    while state["pending"]:
        batch = state["pending"][0]
        if not resolve_device_spans(batch):
            break
        state["pending"].pop(0)
        _append_json(state["path"], batch)


def _layer_from_name(name):
    match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", name)
    return int(match.group(1)) if match else None


def _new_route_counts():
    return {mode: [[0] * 512 for _ in range(48)] for mode in ("extend", "decode")}


def resolve_dispatch_output(args, kwargs):
    """Resolve KTEP's positional or keyword ``dispatch_output`` call form."""
    value = kwargs.get("dispatch_output")
    if value is None:
        if len(args) < 2:
            raise TypeError("Round13 expert observer cannot resolve dispatch_output")
        value = args[-1]
    return value


def _route_document(state):
    totals = {
        mode: sum(sum(layer) for layer in rows)
        for mode, rows in state["route_counts"].items()
    }
    return {
        "schema": 1,
        "status": "ROUND13_FRESH_DEV_ROUTE_COUNTS",
        "intrusive": True,
        "request_prefix": DEV_PREFIX,
        "counts": state["route_counts"],
        "assignment_totals": totals,
        "requests": sorted(state["route_requests"]),
        "copy_scope": "DEV_ONLY_TOPK_IDS_D2H",
    }


def _bind(runner):
    from .candidate_config import load_candidate_profile
    from sglang.srt.layers.moe.kt_ep_wrapper import KTEPWrapperMethod

    candidate, unused, unused2 = load_candidate_profile(
        os.environ.get("ROUND9_CANDIDATE_PROFILE"))
    validate_scope({
        "storage": candidate.storage,
        "placement": candidate.placement,
        "capacity": runner.server_args.kt_num_gpu_experts,
        "workers": runner.server_args.kt_cpuinfer,
        "device": runner.server_args.device,
        "revision": Path(runner.server_args.model_path).name,
    })
    output = Path(os.environ.get("ROUND13_ATTRIBUTION_SPANS", "/out/round13-spans.jsonl"))
    bindings_path = Path(os.environ.get("ROUND13_ATTRIBUTION_BINDINGS", "/out/round13-bindings.json"))
    route_path = Path(os.environ.get("ROUND13_DEV_ROUTE_COUNTS", "/out/round13-dev-route-counts.json"))
    if output.exists() or bindings_path.exists() or route_path.exists():
        raise FileExistsError("Round13 observer outputs must be fresh")

    import torch

    state = {
        "path": output,
        "bindings_path": bindings_path,
        "route_path": route_path,
        "current": None,
        "pending": [],
        "sequences": {},
        "bindings": {name: [] for name in EXPECTED_BINDINGS},
        "route_counts": _new_route_counts(),
        "route_requests": set(),
        "lock": threading.RLock(),
    }
    runner._round13_attribution = state

    def context():
        batch = state["current"]
        if batch is None:
            return None
        return {
            "request_id": batch["request_id"],
            "mode": batch["mode"],
            "batch_sequence": batch["batch_sequence"],
        }

    def note(error):
        batch = state["current"]
        if batch is not None and len(batch["observer_errors"]) < 8:
            batch["observer_errors"].append(error)

    def sink(row):
        with state["lock"]:
            batch = state["current"]
            if batch is None:
                raise RuntimeError("Round13 span escaped its scheduler batch")
            if len(batch["spans"]) >= MAX_SPANS_PER_BATCH:
                batch["dropped_spans"] += 1
            else:
                batch["spans"].append(row)

    def device_pair():
        return torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)

    def wrap(obj, attribute, component, layer, *, device):
        original = getattr(obj, attribute)
        if getattr(original, "_round13_observer", False):
            raise RuntimeError("Round13 provider already observed")
        wrapped = observe(original, sink, component, layer, context,
                          device_pair if device else None, note)
        wrapped._round13_observer = True
        setattr(obj, attribute, wrapped)

    def wrap_expert(method, layer):
        original = method.apply
        observed = observe(original, sink, "ROUTED_EXPERT", layer, context,
                           device_pair, note)

        @functools.wraps(original)
        def call(*args, **kwargs):
            dispatch_output = resolve_dispatch_output(args, kwargs)
            ctx = context()
            if ctx is not None and ctx["request_id"].startswith(DEV_PREFIX):
                ids = dispatch_output.topk_output.topk_ids.detach().cpu().reshape(-1).tolist()
                with state["lock"]:
                    state["route_requests"].add(ctx["request_id"])
                    counts = state["route_counts"][ctx["mode"]][layer]
                    for expert in ids:
                        expert = int(expert)
                        if not 0 <= expert < 512:
                            raise RuntimeError("DEV route expert ID is out of range")
                        counts[expert] += 1
            return observed(*args, **kwargs)

        call._round13_observer = True
        method.apply = call
        wrap(method, "_submit_cpu_forward", "CPU_SUBMIT", layer, device=False)
        wrap(method, "_sync_cpu_forward", "CPU_RESULT_WAIT", layer, device=False)
        wrap(method.gpu_method, "apply", "NPU_RESIDENT", layer, device=True)

    for name, module in runner.model.named_modules():
        layer = _layer_from_name(name)
        kind = type(module).__name__
        if kind == "Qwen4ReferenceGatedDeltaNet" and layer in GDN_LAYERS:
            wrap(module, "forward", "GDN", layer, device=True)
            state["bindings"]["GDN"].append(layer)
        elif kind == "Qwen4ReferenceIndexer" and layer in QSA_LAYERS:
            wrap(module, "forward", "QSA_INDEXER", layer, device=True)
            state["bindings"]["QSA_INDEXER"].append(layer)
        elif kind == "Qwen4ExpAttentionDecoderLayer" and layer in QSA_LAYERS:
            wrap(module, "self_attention", "QSA_ATTENTION", layer, device=True)
            state["bindings"]["QSA_ATTENTION"].append(layer)
        elif kind == "Qwen4ReferencePLE" and int(getattr(module, "layer_id", -1)) == 1:
            wrap(module, "prepare", "PLE_PREPARE", 1, device=False)
            wrap(module, "run", "PLE_CONSUMER", 1, device=True)
            state["bindings"]["PLE"].append(1)
        method = getattr(module, "quant_method", None)
        if isinstance(method, KTEPWrapperMethod) and layer in EXPERT_LAYERS:
            wrap_expert(method, layer)
            state["bindings"]["EXPERT"].append(layer)

    runtime = getattr(runner, "_round9_ple", None)
    if runtime is None:
        raise RuntimeError("Round13 observer requires qualified async PLE")
    wrap(runtime.ring, "wait", "PLE_EXPOSED_WAIT", 1, device=False)
    normalized = {name: sorted(set(values)) for name, values in state["bindings"].items()}
    if normalized != EXPECTED_BINDINGS:
        raise RuntimeError("Round13 provider census differs from frozen sample")
    bindings_path.write_text(json.dumps({
        "schema": 1,
        "status": "ROUND13_P0_ATTRIBUTION_BOUND",
        "bindings": normalized,
        "max_spans_per_batch": MAX_SPANS_PER_BATCH,
        "device_query": "DEFERRED_QUERY_ONLY_NO_SYNCHRONIZE",
        "dev_route_capture": "DEV_ONLY_INTRUSIVE",
    }, indent=2) + "\n")
    return state


def before(scheduler, batch):
    if os.environ.get("ROUND13_ATTRIBUTION_ENABLE") != "1":
        return
    runner = scheduler.tp_worker.model_runner
    state = getattr(runner, "_round13_attribution", None) or _bind(runner)
    with state["lock"]:
        _flush_ready(state)
        if state["current"] is not None or len(batch.reqs) != 1:
            raise RuntimeError("Round13 attribution requires one drained serial batch")
        request = batch.reqs[0]
        mode = ("extend" if batch.forward_mode.is_extend() else
                "decode" if batch.forward_mode.is_decode() else "other")
        if mode == "other":
            raise RuntimeError("Round13 attribution rejects unsupported forward mode")
        sequence = state["sequences"].get(request.rid, 0)
        state["sequences"][request.rid] = sequence + 1
        state["current"] = {
            "schema": 1,
            "status": "ROUND13_P0_BATCH",
            "request_id": request.rid,
            "mode": mode,
            "batch_sequence": sequence,
            "started_ns": time.monotonic_ns(),
            "spans": [],
            "dropped_spans": 0,
            "observer_errors": [],
        }


def after(scheduler):
    if os.environ.get("ROUND13_ATTRIBUTION_ENABLE") != "1":
        return
    state = getattr(scheduler.tp_worker.model_runner, "_round13_attribution", None)
    if state is None:
        raise RuntimeError("Round13 observer was not bound")
    with state["lock"]:
        batch = state["current"]
        if batch is None:
            raise RuntimeError("Round13 scheduler batch is unleased")
        batch["ended_ns"] = time.monotonic_ns()
        batch["duration_ns"] = batch["ended_ns"] - batch["started_ns"]
        state["pending"].append(batch)
        state["current"] = None
        _flush_ready(state)
        if batch["request_id"].startswith(DEV_PREFIX):
            _atomic_json(state["route_path"], _route_document(state))


__all__ = (
    "before", "after", "observe", "resolve_device_spans", "validate_scope",
    "resolve_dispatch_output",
    "GDN_LAYERS", "QSA_LAYERS", "EXPERT_LAYERS", "EXPECTED_BINDINGS",
)
