"""Explicit Round16-B vLLM-Ascend worker; used only with --worker-cls."""

from __future__ import annotations

import faulthandler
import json
import os
from pathlib import Path
import signal
import sys
import time
from typing import Any

from vllm_ascend.worker.worker import NPUWorker


def _install_weight_trace() -> None:
    """Trace the exact fused-linear copy that can block model loading.

    This hook is opt-in and diagnostic only.  It records metadata before the
    native ``copy_`` call, so a process stuck below Python still leaves the
    destination module, shard and tensor layout in the evidence directory.
    """

    raw_path = os.environ.get("ROUND16_B_WEIGHT_TRACE_PATH")
    if not raw_path:
        return
    path = Path(raw_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    from vllm.model_executor.layers.linear import MergedColumnParallelLinear

    original = MergedColumnParallelLinear.weight_loader
    if getattr(original, "round16_weight_trace", False):
        return

    def emit(payload: dict[str, Any]) -> None:
        payload = {"monotonic": time.monotonic(), **payload}
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def describe(value: Any) -> dict[str, Any]:
        return {
            "shape": list(getattr(value, "shape", ())),
            "stride": list(value.stride()) if callable(getattr(value, "stride", None)) else None,
            "dtype": str(getattr(value, "dtype", None)),
            "device": str(getattr(value, "device", None)),
            "contiguous": bool(value.is_contiguous())
            if callable(getattr(value, "is_contiguous", None))
            else None,
        }

    def traced(
        self: Any,
        param: Any,
        loaded_weight: Any,
        loaded_shard_id: tuple[int, ...] | int | None = None,
    ) -> Any:
        sequence = int(getattr(self, "round16_weight_trace_sequence", 0)) + 1
        self.round16_weight_trace_sequence = sequence
        common = {
            "sequence": sequence,
            "prefix": str(getattr(self, "prefix", "")),
            "shard": loaded_shard_id,
            "param": describe(getattr(param, "data", param)),
            "loaded": describe(loaded_weight),
        }
        emit({"event": "begin", **common})
        started = time.monotonic()
        result = original(self, param, loaded_weight, loaded_shard_id)
        emit(
            {
                "event": "end",
                "elapsed_seconds": time.monotonic() - started,
                **common,
            }
        )
        return result

    traced.round16_weight_trace = True
    MergedColumnParallelLinear.weight_loader = traced
    emit({"event": "installed", "class": "MergedColumnParallelLinear"})


class Round16BHybridWorker(NPUWorker):
    """Install the default-OFF B2 model hook immediately before model load."""

    def load_model(self) -> None:
        # This executes inside EngineCore after the frozen runtime creates the
        # worker process.  The parent probe's signal handler is not retained by
        # this launch path, so register the bounded diagnostic hook here too.
        faulthandler.register(signal.SIGUSR2, file=sys.stderr, all_threads=True)
        _install_weight_trace()
        from round16_integration.b2_worker_overlay_entry import (
            maybe_install_from_environment,
        )

        installed = maybe_install_from_environment(type(self.model_runner))
        if installed is None:
            raise RuntimeError(
                "Round16BHybridWorker requires explicit ROUND16_B_ENABLE_B2=1"
            )
        self.round16_b_installed_plugin = installed
        return super().load_model()


__all__ = ("Round16BHybridWorker",)
