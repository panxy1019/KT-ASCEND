#!/usr/bin/env python3
"""Build helpers for the Round13 direct BF16 QSA Decode candidate.

The candidate changes one ownership decision in the already promoted QSA
runtime.  Qualified BF16 Decode consumes the resident device KV tensors
directly.  Prefill keeps the Round12 vectorized branch and every unqualified
shape keeps the original staged-row fallback.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


BASELINE_RUNTIME_SHA256 = (
    "0b6e6a7fad90e3856ffdcbea233a13baaf1bb342837c68ff344579370773c028"
)


def _replace_once(source: str, old: str, new: str, label: str) -> str:
    if source.count(old) != 1:
        raise ValueError(f"{label} precondition mismatch")
    return source.replace(old, new)


def patch_context_runtime(source: str) -> str:
    """Enable direct resident-BF16 Decode for the exact TP1 text geometry."""
    old = '''    qualified = (
        mode == "BF16_COMPACT"
        and q.ndim == 3 and tuple(q.shape[1:]) == (24, 256)
        and q.shape[0] > 1 and q.dtype == torch.bfloat16
        and keys.ndim == 3 and tuple(keys.shape[1:]) == (2, 256)
        and values.shape == keys.shape
        and keys.dtype == torch.bfloat16 and values.dtype == torch.bfloat16
        and selected.ndim == 2 and selected.shape[0] == q.shape[0]
        and selected.dtype in (torch.int32, torch.int64)
        and positions.ndim == 1 and positions.shape[0] == q.shape[0]
        and positions.dtype in (torch.int32, torch.int64)
    )
    if not qualified:
        result = _consume_reference_rows(
            identity, mode, layer_id, q, keys, values, selected, positions,
            reference_consumer)
        return result, "ROW" if q.shape[0] == 1 else "ROW_FALLBACK"
    valid = selected >= 0
    if bool((valid & (selected > positions[:, None])).any()):
        raise ValueError("QSA candidate selected a future logical token")
    result = reference_consumer(q, keys, values, selected)
    if result.shape != q.shape or result.dtype != q.dtype:
        raise RuntimeError("QSA candidate changed output shape or dtype")
    return result, "VECTORIZED"
'''
    new = '''    qualified_common = (
        mode == "BF16_COMPACT"
        and q.ndim == 3 and tuple(q.shape[1:]) == (24, 256)
        and q.dtype == torch.bfloat16
        and keys.ndim == 3 and tuple(keys.shape[1:]) == (2, 256)
        and values.shape == keys.shape
        and keys.dtype == torch.bfloat16 and values.dtype == torch.bfloat16
        and selected.ndim == 2 and selected.shape[0] == q.shape[0]
        and selected.dtype in (torch.int32, torch.int64)
        and positions.ndim == 1 and positions.shape[0] == q.shape[0]
        and positions.dtype in (torch.int32, torch.int64)
    )
    qualified_prefill = qualified_common and q.shape[0] > 1
    qualified_decode = qualified_common and q.shape[0] == 1
    if not (qualified_prefill or qualified_decode):
        result = _consume_reference_rows(
            identity, mode, layer_id, q, keys, values, selected, positions,
            reference_consumer)
        return result, "ROW" if q.shape[0] == 1 else "ROW_FALLBACK"
    valid = selected >= 0
    if bool((valid & (selected > positions[:, None])).any()):
        raise ValueError("QSA candidate selected a future logical token")
    result = reference_consumer(q, keys, values, selected)
    if result.shape != q.shape or result.dtype != q.dtype:
        raise RuntimeError("QSA candidate changed output shape or dtype")
    strategy = "VECTORIZED" if qualified_prefill else "DIRECT_BF16_ROW"
    return result, strategy
'''
    patched = _replace_once(source, old, new, "Round13 QSA Decode runtime")
    compile(patched, "context_reference_runtime.py", "exec")
    return patched


def bind_candidate_runtime(root: Path, candidate_runtime_sha256: str) -> dict:
    """Bind the existing promoted-QSA proof to one exact runtime digest."""
    root = Path(root)
    if (
        len(candidate_runtime_sha256) != 64
        or any(character not in "0123456789abcdef" for character in candidate_runtime_sha256)
        or candidate_runtime_sha256 == BASELINE_RUNTIME_SHA256
    ):
        raise ValueError("invalid Round13 B runtime digest")
    binding_path = root / "ROUND12_QSA_SOURCE_BINDING.json"
    original_raw = binding_path.read_bytes()
    binding = json.loads(original_raw)
    if binding.get("candidate_context_reference_runtime_sha256") != BASELINE_RUNTIME_SHA256:
        raise ValueError("Round13 B parent runtime identity mismatch")
    if binding.get("candidate") != "QUERY_AWARE_VECTORIZED_SPARSE_GQA":
        raise ValueError("Round13 B parent QSA identity mismatch")
    gate_digests = {}
    for gate_path in (
        root / "source/runtime/context_reference_gate.py",
        root / "overlay-python/sglang/srt/layers/round9/context_reference_gate.py",
    ):
        gate = _replace_once(
            gate_path.read_text(),
            BASELINE_RUNTIME_SHA256,
            candidate_runtime_sha256,
            f"Round13 B exact runtime tuple {gate_path}",
        )
        compile(gate, str(gate_path), "exec")
        gate_path.write_text(gate)
        gate_digests[str(gate_path.relative_to(root))] = hashlib.sha256(
            gate.encode()
        ).hexdigest()
    binding["candidate_context_reference_runtime_sha256"] = candidate_runtime_sha256
    binding_path.write_text(json.dumps(binding, indent=2) + "\n")
    delta = {
        "schema": 1,
        "status": "PASS_ROUND13_B_DIRECT_BF16_DECODE_SOURCE_BINDING",
        "candidate": "DIRECT_RESIDENT_BF16_ROW_DECODE",
        "baseline_binding_sha256": hashlib.sha256(original_raw).hexdigest(),
        "baseline_runtime_sha256": BASELINE_RUNTIME_SHA256,
        "candidate_runtime_sha256": candidate_runtime_sha256,
        "unchanged_qsa_sha256": binding["candidate_qsa_sha256"],
        "unchanged_qwen4_exp_sha256": binding["candidate_qwen4_exp_sha256"],
        "gate_sha256": gate_digests,
        "scope": "ONE_RUNTIME_OWNERSHIP_CHANGE_DIRECT_DEVICE_BF16_DECODE",
    }
    (root / "ROUND13_B_QSA_SOURCE_DELTA.json").write_text(
        json.dumps(delta, indent=2) + "\n"
    )
    return delta


__all__ = ("patch_context_runtime", "bind_candidate_runtime")
