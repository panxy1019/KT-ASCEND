"""Default-off Round11 GDN/QSA/PLE observer with deferred NPU event queries.

The observer samples real early/middle/late GDN and QSA layers plus the sole
PLE layer.  It never synchronizes a device event.  An incomplete event pair is
held until a later existing scheduler boundary reports ``query() == True``.
"""

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
MAX_SPANS_PER_BATCH = 64
EXPECTED_BINDINGS = {
    "GDN": list(GDN_LAYERS),
    "QSA_ATTENTION": list(QSA_LAYERS),
    "QSA_INDEXER": list(QSA_LAYERS),
    "PLE": [1],
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
        raise ValueError("Round11 B observer outside frozen static Hybrid scope")


def observe(original, sink, component, layer, context, device_factory=None,
            on_error=None, clock=time.monotonic_ns):
    """Observe one provider without changing its return value or exception."""
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
                "device_status": ("PENDING" if pair is not None else
                                  "NOT_APPLICABLE" if device_factory is None else
                                  "OBSERVER_ERROR"),
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
    """Query completed event pairs; return False while any pair remains pending."""
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


def _bind(runner):
    from .candidate_config import load_candidate_profile

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
    output = Path(os.environ.get("ROUND11_B_SPANS", "/out/round11-b-spans.jsonl"))
    bindings_path = Path(os.environ.get(
        "ROUND11_B_BINDINGS", "/out/round11-b-bindings.json"))
    if output.exists() or bindings_path.exists():
        raise FileExistsError("Round11 B observer outputs must be fresh")

    import torch

    state = {
        "path": output,
        "bindings_path": bindings_path,
        "current": None,
        "pending": [],
        "sequences": {},
        "bindings": {name: [] for name in EXPECTED_BINDINGS},
        "lock": threading.RLock(),
    }
    runner._round11_b_attribution = state

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
                raise RuntimeError("Round11 B span escaped its scheduler batch")
            if len(batch["spans"]) >= MAX_SPANS_PER_BATCH:
                batch["dropped_spans"] += 1
            else:
                batch["spans"].append(row)

    def device_pair():
        return torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)

    def wrap(obj, attribute, component, layer, *, device):
        original = getattr(obj, attribute)
        if getattr(original, "_round11_b_observer", False):
            raise RuntimeError("Round11 B provider already observed")
        wrapped = observe(original, sink, component, layer, context,
                          device_pair if device else None, note)
        wrapped._round11_b_observer = True
        setattr(obj, attribute, wrapped)

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

    runtime = getattr(runner, "_round9_ple", None)
    if runtime is None:
        raise RuntimeError("Round11 B observer requires the qualified async PLE runtime")
    wrap(runtime.ring, "wait", "PLE_EXPOSED_WAIT", 1, device=False)
    normalized = {name: sorted(values) for name, values in state["bindings"].items()}
    if normalized != EXPECTED_BINDINGS:
        raise RuntimeError("Round11 B provider census differs from the frozen sample")
    bindings_path.write_text(json.dumps({
        "schema": 1,
        "status": "ROUND11_B_ATTRIBUTION_BOUND",
        "bindings": normalized,
        "max_spans_per_batch": MAX_SPANS_PER_BATCH,
        "device_query": "DEFERRED_QUERY_ONLY_NO_SYNCHRONIZE",
    }, indent=2) + "\n")
    return state


def before(scheduler, batch):
    if os.environ.get("ROUND11_B_ATTRIBUTION_ENABLE") != "1":
        return
    runner = scheduler.tp_worker.model_runner
    state = getattr(runner, "_round11_b_attribution", None) or _bind(runner)
    with state["lock"]:
        _flush_ready(state)
        if state["current"] is not None or len(batch.reqs) != 1:
            raise RuntimeError("Round11 B attribution requires one drained serial batch")
        request = batch.reqs[0]
        mode = ("extend" if batch.forward_mode.is_extend() else
                "decode" if batch.forward_mode.is_decode() else "other")
        if mode == "other":
            raise RuntimeError("Round11 B attribution rejects unsupported forward mode")
        sequence = state["sequences"].get(request.rid, 0)
        state["sequences"][request.rid] = sequence + 1
        state["current"] = {
            "schema": 1,
            "status": "ROUND11_B_BATCH",
            "request_id": request.rid,
            "mode": mode,
            "batch_sequence": sequence,
            "started_ns": time.monotonic_ns(),
            "spans": [],
            "dropped_spans": 0,
            "observer_errors": [],
        }


def after(scheduler):
    if os.environ.get("ROUND11_B_ATTRIBUTION_ENABLE") != "1":
        return
    state = getattr(scheduler.tp_worker.model_runner, "_round11_b_attribution", None)
    if state is None:
        raise RuntimeError("Round11 B observer was not bound")
    with state["lock"]:
        batch = state["current"]
        if batch is None:
            raise RuntimeError("Round11 B scheduler batch is unleased")
        batch["ended_ns"] = time.monotonic_ns()
        batch["duration_ns"] = batch["ended_ns"] - batch["started_ns"]
        state["pending"].append(batch)
        state["current"] = None
        _flush_ready(state)


__all__ = (
    "before", "after", "observe", "resolve_device_spans", "validate_scope",
    "GDN_LAYERS", "QSA_LAYERS", "EXPECTED_BINDINGS",
)
