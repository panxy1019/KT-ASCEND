"""Default-off QSA compact/FP8 host reference consumer and batch lifecycle."""

import copy
import hashlib
import json
import os
from pathlib import Path
import re

from .context_profile import ContextStateIdentity


MODES = ("BF16_COMPACT", "FP8_HOST_REFERENCE")
QSA_LAYERS = tuple(range(3, 48, 4))
SCOPE = "MODEL_REFERENCE_CONSUMER_NO_HBM_RELEASE"


def parse_context_reference_profile(document):
    if (not isinstance(document, dict)
            or set(document) != {"schema", "mode", "device_gate"}
            or type(document["schema"]) is not int or document["schema"] != 1
            or document["mode"] not in MODES):
        raise ValueError("invalid context reference profile")
    gate = document["device_gate"]
    if (not isinstance(gate, dict) or set(gate) != {"path", "sha256"}
            or not isinstance(gate["path"], str) or not Path(gate["path"]).is_absolute()
            or not isinstance(gate["sha256"], str)
            or re.fullmatch("[0-9a-f]{64}", gate["sha256"]) is None):
        raise ValueError("context reference requires a frozen absolute device receipt")
    return copy.deepcopy(document)


def load_context_reference_profile():
    path = os.environ.get("ROUND9_CONTEXT_REFERENCE_PROFILE")
    expected = os.environ.get("ROUND9_CONTEXT_REFERENCE_SHA256")
    if not path and not expected:
        return None
    if not path or not expected or not Path(path).is_absolute():
        raise ValueError("context reference profile and SHA are required together")
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("context reference profile changed")
    return parse_context_reference_profile(json.loads(raw))


def _consume_reference_rows(identity, mode, layer_id, q, keys, values, selected,
                            positions, reference_consumer):
    import torch
    from .context_staging import ResidentBF16Page, stage_selected_rows
    from .context_page_store import PageKey, PAGE_SIZE
    from .kv_quant_reference import encode_kv_page

    if (not isinstance(q, torch.Tensor) or not isinstance(keys, torch.Tensor)
            or not isinstance(values, torch.Tensor) or not isinstance(selected, torch.Tensor)
            or not isinstance(positions, torch.Tensor)):
        raise TypeError("QSA reference consumer requires Torch tensors")
    if (keys.dtype != torch.bfloat16 or values.dtype != torch.bfloat16
            or tuple(keys.shape) != tuple(values.shape)
            or keys.ndim != 3 or tuple(keys.shape[1:]) != (2, 256)
            or selected.ndim != 2 or selected.shape[0] != q.shape[0]
            or positions.numel() != q.shape[0]):
        raise ValueError("QSA reference consumer geometry differs from TP1 contract")
    logical_positions = positions.detach().cpu().reshape(-1).tolist()
    outputs = []
    for row, current in enumerate(logical_positions):
        current = int(current)
        row_selected = selected[row:row + 1].detach().cpu()

        def resident_reader(page_key, requested_layer):
            start = page_key.logical_page * PAGE_SIZE
            stop = min(start + PAGE_SIZE, keys.shape[0])
            if stop <= start:
                raise ValueError("resident logical page exceeds live KV history")
            return ResidentBF16Page(
                page_key, requested_layer,
                keys[start:stop].detach().cpu().contiguous(),
                values[start:stop].detach().cpu().contiguous(),
            )

        def record_reader(page_key, requested_layer):
            resident = resident_reader(page_key, requested_layer)
            return encode_kv_page(
                identity, page_key.logical_page, requested_layer,
                resident.keys, resident.values,
            )

        staged = stage_selected_rows(
            identity, layer_id, row_selected,
            current_logical_position=current,
            record_reader=record_reader if mode == "FP8_HOST_REFERENCE" else None,
            resident_reader=resident_reader if mode == "BF16_COMPACT" else None,
        )
        device = q.device
        staged_keys = staged.keys.to(device=device)
        staged_values = staged.values.to(device=device)
        remapped = staged.remapped.to(device=device)
        outputs.append(reference_consumer(
            q[row:row + 1], staged_keys, staged_values, remapped))
    return torch.cat(outputs, dim=0)


class ContextReferenceRuntime:
    """One serial request; each scheduler batch must consume all QSA layers."""

    def __init__(self, mode, *, layers=QSA_LAYERS, consume_rows=None, event=None):
        if mode not in MODES or not layers or any(type(layer) is not int for layer in layers):
            raise ValueError("invalid context reference runtime geometry")
        self.mode = mode
        self.layers = tuple(layers)
        self.consume_rows = consume_rows or _consume_reference_rows
        self.event = event or (lambda item: None)
        self.state = "READY"
        self.identity = None
        self.request_token = None
        self._next_epoch = 0
        self.active = False
        self.next_layer = 0

    def _ready(self):
        if self.state != "READY":
            raise RuntimeError(self.state)

    def _poison(self, error):
        self.state = "BROKEN_RELOAD_REQUIRED"
        self.active = False
        raise error

    def begin(self, request_id, request_token):
        self._ready()
        if self.active:
            raise RuntimeError("a context reference batch is already active")
        if self.request_token is None:
            self.identity = ContextStateIdentity(
                request_id, None, self._next_epoch, "QSA_KV")
            self._next_epoch += 1
            self.request_token = request_token
        elif request_token is not self.request_token or request_id != self.identity.request_id:
            raise RuntimeError("another context request is still active")
        self.active = True
        self.next_layer = 0

    def consume(self, layer_id, q, keys, values, selected, positions, reference_consumer):
        self._ready()
        if (not self.active or self.next_layer >= len(self.layers)
                or layer_id != self.layers[self.next_layer]):
            self._poison(RuntimeError("QSA layers must consume in exact ordered sequence"))
        try:
            result = self.consume_rows(
                self.identity, self.mode, layer_id, q, keys, values, selected,
                positions, reference_consumer)
            self.next_layer += 1
            return result
        except BaseException as error:
            self._poison(error)

    def after(self, request_token, *, finished):
        self._ready()
        if request_token is not self.request_token or not self.active:
            self._poison(RuntimeError("context request identity is stale or unleased"))
        if self.next_layer != len(self.layers):
            self._poison(RuntimeError("context batch did not consume all QSA layers"))
        self.active = False
        try:
            self.event({
                "status": "CONTEXT_REFERENCE_BATCH_DRAINED",
                "request_id": self.identity.request_id,
                "allocation_epoch": self.identity.allocation_epoch,
                "mode": self.mode,
                "qsa_layers": len(self.layers),
                "hbm_release": False,
            })
            if finished:
                self.event({
                    "status": "CONTEXT_REFERENCE_REQUEST_DRAINED",
                    "request_id": self.identity.request_id,
                    "allocation_epoch": self.identity.allocation_epoch,
                    "hbm_release": False,
                })
                self.identity = self.request_token = None
        except BaseException as error:
            self._poison(error)

    def abort(self, request_token):
        self._ready()
        if request_token is not self.request_token:
            raise RuntimeError("context abort identity is stale")
        identity = self.identity
        self.active = False
        self.identity = self.request_token = None
        self.event({
            "status": "CONTEXT_REFERENCE_ABORTED",
            "request_id": identity.request_id,
            "allocation_epoch": identity.allocation_epoch,
            "hbm_release": False,
        })


__all__ = (
    "ContextReferenceRuntime", "parse_context_reference_profile",
    "load_context_reference_profile", "MODES", "QSA_LAYERS", "SCOPE",
)
