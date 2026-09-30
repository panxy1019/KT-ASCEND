"""Bounded diagnostic capture of the exact native Top-K routed occurrences."""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
from threading import Lock
import time
from typing import Any


_LAYER_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


def _host_values(tensor: Any) -> Any:
    value = tensor
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    tolist = getattr(value, "tolist", None)
    if not callable(tolist):
        raise TypeError("route tensor does not support tolist")
    return tolist()


def _matrix_shape(values: Any, *, name: str) -> tuple[int, int]:
    if not isinstance(values, list):
        raise RuntimeError(f"{name} is not a row matrix")
    if not values:
        return (0, 0)
    if any(not isinstance(row, list) for row in values):
        raise RuntimeError(f"{name} is not a row matrix")
    widths = {len(row) for row in values}
    if len(widths) != 1:
        raise RuntimeError(f"{name} rows have inconsistent widths")
    return (len(values), widths.pop())


class NativeRouteTraceRecorder:
    """Append exact Top-K tensors only while an explicit enable marker exists.

    The recorder is a diagnostic observer.  It intentionally performs D2H and
    file I/O and therefore its run cannot be used as a performance sample.
    """

    def __init__(
        self,
        path: Path,
        *,
        max_events: int = 20_000,
        enabled_path: Path | None = None,
    ) -> None:
        if type(max_events) is not int or max_events <= 0:
            raise ValueError("max_events must be positive")
        self.path = Path(path)
        self.enabled_path = Path(enabled_path) if enabled_path is not None else None
        self.max_events = max_events
        self._event_index = 0
        self._lock = Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # A new immutable run must never append to an older route identity.
        with self.path.open("x", encoding="utf-8"):
            pass

    @property
    def event_count(self) -> int:
        return self._event_index

    def record(
        self,
        *,
        layer_name: str,
        generation: int,
        hidden_states: Any,
        topk_ids: Any,
        topk_weights: Any,
        identity_key: tuple[Any, ...],
    ) -> None:
        if self.enabled_path is not None and not self.enabled_path.is_file():
            return
        with self._lock:
            if self._event_index >= self.max_events:
                raise RuntimeError("native route trace event limit exceeded")
            match = _LAYER_PATTERN.search(layer_name)
            if match is None:
                raise RuntimeError(f"cannot parse logical layer from {layer_name!r}")
            ids = _host_values(topk_ids)
            weights = _host_values(topk_weights)
            id_shape = _matrix_shape(ids, name="topk_ids")
            weight_shape = _matrix_shape(weights, name="topk_weights")
            if id_shape != weight_shape:
                raise RuntimeError(
                    f"route tensor shape mismatch: ids={id_shape}, weights={weight_shape}"
                )
            hidden_shape = tuple(int(value) for value in getattr(hidden_states, "shape"))
            if not hidden_shape or hidden_shape[0] != id_shape[0]:
                raise RuntimeError(
                    "hidden/route row mismatch: "
                    f"hidden={hidden_shape}, route={id_shape}"
                )
            if any(
                not math.isfinite(float(weight))
                for row in weights
                for weight in row
            ):
                raise RuntimeError("route trace contains a non-finite Top-K weight")
            event = {
                "schema": 1,
                "kind": "native_route_event",
                "event_index": self._event_index,
                "monotonic_ns": time.monotonic_ns(),
                "layer_name": layer_name,
                "logical_layer": int(match.group(1)),
                "generation": int(generation),
                "identity_repr": repr(tuple(identity_key)),
                "token_rows": id_shape[0],
                "top_k": id_shape[1],
                "hidden_shape": list(hidden_shape),
                "topk_ids": ids,
                "topk_weights": weights,
            }
            raw = json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n"
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(raw)
                stream.flush()
            self._event_index += 1


__all__ = ("NativeRouteTraceRecorder",)
