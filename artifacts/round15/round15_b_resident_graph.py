"""Exact fixed-buffer NPUGraph wrapper for one resident MoE Decode segment."""

from __future__ import annotations

import gc
import atexit
import json
import os
from pathlib import Path
import threading
import time

import torch

try:
    from .round15_b_segment_policy import decide_resident_moe_graph
except ImportError:  # Standalone component snapshot before package integration.
    from round15_b_segment_policy import decide_resident_moe_graph


_REGISTRY = []
_REGISTRY_LOCK = threading.Lock()
_MAX_GRAPHS = 48


def _write_event(row):
    path = os.environ.get("ROUND15_B_GRAPH_EVENTS")
    if not path:
        return
    document = {"time": time.time(), **row}
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(document, sort_keys=True) + "\n")


class ResidentMoeDecodeGraph:
    """Capture only ``gpu_method.apply`` for one immutable resident layer.

    CPU submission, CPU completion, routed merge, QSA/GDN/PLE state, request
    epochs, and cancellation remain outside this object.  The owner must be
    serial and must call ``reset`` during model teardown.
    """

    def __init__(self, method, layer, dispatch_output):
        self.method = method
        self.layer = layer
        self.graph = None
        self.capture_stream = None
        self.static_dispatch = None
        self.static_hidden = None
        self.static_topk_ids = None
        self.static_topk_weights = None
        self.output = None
        self.capture(dispatch_output)

    def _copy_inputs(self, dispatch_output):
        self.static_hidden.copy_(dispatch_output.hidden_states)
        self.static_topk_ids.copy_(dispatch_output.topk_output.topk_ids)
        self.static_topk_weights.copy_(dispatch_output.topk_output.topk_weights)

    def capture(self, dispatch_output):
        if self.graph is not None:
            raise RuntimeError("Round15 resident MoE graph was already captured")
        topk = dispatch_output.topk_output
        self.static_hidden = torch.empty_like(dispatch_output.hidden_states)
        self.static_topk_ids = torch.empty_like(topk.topk_ids)
        self.static_topk_weights = torch.empty_like(topk.topk_weights)
        self._copy_inputs(dispatch_output)
        static_topk = topk._replace(
            topk_ids=self.static_topk_ids,
            topk_weights=self.static_topk_weights,
        )
        self.static_dispatch = dispatch_output._replace(
            hidden_states=self.static_hidden,
            topk_output=static_topk,
        )
        self.capture_stream = torch.npu.Stream()
        current = torch.npu.current_stream(self.static_hidden.device)
        self.capture_stream.wait_stream(current)
        with torch.npu.stream(self.capture_stream):
            for _ in range(3):
                self.method.apply(self.layer, self.static_dispatch)
        current.wait_stream(self.capture_stream)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        try:
            with torch.npu.graph(
                graph,
                stream=self.capture_stream,
                auto_dispatch_capture=True,
            ):
                output = self.method.apply(self.layer, self.static_dispatch)
            current.wait_stream(self.capture_stream)
            torch.npu.synchronize()
        except BaseException:
            try:
                graph.reset()
            finally:
                raise
        self.graph = graph
        self.output = output

    def replay(self, dispatch_output):
        if self.graph is None:
            raise RuntimeError("Round15 resident MoE graph is closed")
        self._copy_inputs(dispatch_output)
        self.graph.replay()
        return self.output

    def reset(self):
        graph, self.graph = self.graph, None
        if graph is not None:
            graph.reset()
        self.output = None
        self.static_dispatch = None
        self.static_hidden = None
        self.static_topk_ids = None
        self.static_topk_weights = None
        self.capture_stream = None
        gc.collect()
        if torch.npu.is_available():
            torch.npu.synchronize()


def _register_graph(graph):
    with _REGISTRY_LOCK:
        if len(_REGISTRY) >= _MAX_GRAPHS:
            raise RuntimeError("Round15 resident MoE graph registry exceeded 48 layers")
        _REGISTRY.append(graph)


def reset_all_graphs():
    with _REGISTRY_LOCK:
        graphs = list(reversed(_REGISTRY))
        _REGISTRY.clear()
    reset = 0
    for graph in graphs:
        try:
            graph.reset()
            reset += 1
        except BaseException:
            pass
    if reset:
        _write_event({"event": "PROCESS_EXIT_RESET", "graphs": reset})


atexit.register(reset_all_graphs)


def apply_resident_moe_graph(context, layer, dispatch_output):
    """Return ``(used, combine_input)`` for the exact qualified segment."""

    from sglang.srt.server_args import get_global_server_args

    topk = dispatch_output.topk_output
    args = get_global_server_args()
    decision = decide_resident_moe_graph(
        enabled=os.environ.get("ROUND15_B_RESIDENT_GRAPH_ENABLE") == "1",
        forward_mode=getattr(context, "_round15_forward_mode", None),
        hidden_shape=dispatch_output.hidden_states.shape,
        topk_ids_shape=topk.topk_ids.shape,
        topk_weights_shape=topk.topk_weights.shape,
        hidden_dtype=str(dispatch_output.hidden_states.dtype),
        ids_dtype=str(topk.topk_ids.dtype),
        weights_dtype=str(topk.topk_weights.dtype),
        layer_idx=int(context.kt_config.layer_idx),
        tp_size=int(args.tp_size),
        num_gpu_experts=int(context.num_gpu_experts),
        method_name=type(context.gpu_method).__name__,
        return_fp32_contribution=bool(
            getattr(context.gpu_method, "_kt_return_fp32_contribution", False)
        ),
        with_bias=bool(getattr(context.gpu_method, "with_bias", False)),
        max_running_requests=int(args.max_running_requests or 0),
    )
    if not decision.use_graph:
        reasons = getattr(context, "_round15_graph_fallback_reasons", None)
        if reasons is None:
            reasons = context._round15_graph_fallback_reasons = set()
        if decision.reason not in reasons:
            reasons.add(decision.reason)
            _write_event(
                {
                    "event": "EAGER_FALLBACK",
                    "layer": int(context.kt_config.layer_idx),
                    "reason": decision.reason,
                }
            )
        return False, None

    graph = getattr(context, "_round15_resident_moe_graph", None)
    if graph is None:
        eager = context.gpu_method.apply(layer, dispatch_output)
        torch.npu.synchronize()
        before = int(torch.npu.memory_allocated())
        graph = ResidentMoeDecodeGraph(context.gpu_method, layer, dispatch_output)
        candidate = graph.replay(dispatch_output)
        torch.npu.synchronize()
        eager_value = eager.hidden_states
        candidate_value = candidate.hidden_states
        # The one-time capture proof is control-plane work.  Evaluate it on
        # CPU so the runtime contract does not depend on optional NPU linalg
        # kernels, and keep the measured replay path free of this check.
        eager_check = eager_value.detach().float().cpu()
        candidate_check = candidate_value.detach().float().cpu()
        relative_l2 = float(
            torch.linalg.vector_norm(candidate_check - eager_check)
            / torch.linalg.vector_norm(eager_check).clamp_min(1e-30)
        )
        maximum = float((candidate_check - eager_check).abs().max())
        if relative_l2 > 1e-5 or not bool(torch.isfinite(candidate_check).all()):
            graph.reset()
            raise RuntimeError(
                "Round15 resident MoE graph capture self-check failed: "
                f"relative_l2={relative_l2} max_abs={maximum}"
            )
        _register_graph(graph)
        context._round15_resident_moe_graph = graph
        context._round15_graph_replays = 1
        _write_event(
            {
                "event": "CAPTURE_AND_SELF_CHECK",
                "layer": int(context.kt_config.layer_idx),
                "relative_l2": relative_l2,
                "max_abs": maximum,
                "allocated_delta": int(torch.npu.memory_allocated()) - before,
                "key": list(decision.key),
            }
        )
        return True, candidate

    result = graph.replay(dispatch_output)
    context._round15_graph_replays = int(
        getattr(context, "_round15_graph_replays", 0)
    ) + 1
    count = context._round15_graph_replays
    if count in (2, 16, 64, 128):
        _write_event(
            {
                "event": "REPLAY",
                "layer": int(context.kt_config.layer_idx),
                "count": count,
            }
        )
    return True, result


__all__ = (
    "ResidentMoeDecodeGraph",
    "apply_resident_moe_graph",
    "reset_all_graphs",
)
