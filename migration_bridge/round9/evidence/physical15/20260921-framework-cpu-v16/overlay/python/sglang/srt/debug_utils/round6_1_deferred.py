"""Opt-in tensor capture for the Round 6.1 Deferred root-cause experiment.

This module is inert unless ``KT_R61_STAGEWISE_DIR`` is set.  It is deliberately
small and model-local: the files are diagnostic artifacts, not a serving API.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch


_capture_lock = threading.Lock()
_capture_counts: Dict[Tuple[str, Optional[int], str], int] = {}


def _selected(layer: Optional[int]) -> bool:
    if layer is None:
        return True
    requested = {
        int(value)
        for value in os.environ.get(
            "KT_R61_STAGEWISE_MODEL_LAYERS", "7,8,9,12,17,22,26"
        ).split(",")
        if value.strip()
    }
    return layer in requested


def capture_tensor(
    stage: str,
    tensor: Optional[torch.Tensor],
    *,
    layer: Optional[int] = None,
    source: str,
) -> None:
    if os.getenv("KT_R62_COMPLETE_FORWARD_DUMP") == "1" and tensor is not None:
        from sglang.srt.debug_utils.tensor_dump_forward_hook import record_round62_stage
        record_round62_stage(source, layer, stage, tensor)
    root = os.environ.get("KT_R61_STAGEWISE_DIR")
    if not root or tensor is None or not _selected(layer):
        return
    target = Path(root)
    target.mkdir(parents=True, exist_ok=True)
    prefix = "final" if layer is None else f"layer{layer:02d}"
    payload = {
        "stage": stage,
        "layer": layer,
        "source": source,
        "source_device": str(tensor.device),
        "source_dtype": str(tensor.dtype),
        "tensor": tensor.detach().to(device="cpu").contiguous(),
    }
    key = (source, layer, stage)
    with _capture_lock:
        call_index = _capture_counts.get(key, 0) + 1
        _capture_counts[key] = call_index
    torch.save(
        payload,
        target / f"{source}_{prefix}_{stage}_call{call_index:04d}.pt",
    )
