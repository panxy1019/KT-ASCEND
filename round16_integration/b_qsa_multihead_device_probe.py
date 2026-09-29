"""Real-NPU QSA multi-page, multi-KV-head store and attention probe."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback
from typing import Any


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _tensor_sha256(tensor: Any) -> str:
    contiguous = tensor.detach().to(device="cpu").contiguous()
    raw = contiguous
    # bfloat16 is not universally supported by NumPy.  View its storage as
    # int16 without changing bytes before materializing the bounded fixture.
    if str(contiguous.dtype) == "torch.bfloat16":
        import torch

        raw = contiguous.view(torch.int16)
    return hashlib.sha256(raw.numpy().tobytes()).hexdigest()


def _relative_l2(actual: Any, expected: Any) -> float:
    numerator = (actual.float() - expected.float()).pow(2).sum().sqrt()
    denominator = expected.float().pow(2).sum().sqrt().clamp_min(1e-12)
    return float((numerator / denominator).item())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    started = time.time()
    try:
        import torch
        import torch_npu  # noqa: F401
        from vllm_ascend.models.qwen4_exp import qsa as qsa_module
        from vllm_ascend.models.qwen4_exp.ops import (
            qsa_sparse_paged_attention as reference_attention,
        )
        from vllm_ascend.ops.triton.qwen4_exp.qsa import (
            qsa_sparse_paged_attention as triton_attention,
        )

        from round16_integration.b2_live_bindings import (
            ASCEND_QSA_MULTIHEAD_STORE_API,
            install_ascend_qsa_cache_split,
            install_ascend_qsa_multihead_store,
        )

        torch.npu.set_device(0)
        torch.manual_seed(20260929)
        page_count = 3
        page_size = 3072
        kv_heads = 2
        query_heads = 24
        head_size = 256
        row_count = 12
        device = torch.device("npu:0")

        key_cache = torch.zeros(
            (page_count, page_size, kv_heads, head_size),
            dtype=torch.bfloat16,
            device=device,
        )
        value_cache = torch.zeros_like(key_cache)
        key = torch.randn(
            (row_count, kv_heads, head_size),
            dtype=torch.bfloat16,
            device=device,
        )
        value = torch.randn_like(key)

        # Cross every page boundary and use a non-identity logical-to-physical
        # block table.  Store slots are physical; attention indices are logical.
        logical_positions = torch.tensor(
            [0, 1, 3071, 3072, 3073, 6143, 6144, 6145, 9214, 1500, 4700, 8000],
            dtype=torch.int64,
        )
        block_table_cpu = torch.tensor([[2, 0, 1]], dtype=torch.int32)
        logical_pages = logical_positions // page_size
        offsets = logical_positions % page_size
        physical_pages = block_table_cpu[0, logical_pages]
        slot_mapping_cpu = physical_pages.to(torch.int64) * page_size + offsets
        slot_mapping = slot_mapping_cpu.to(device=device)

        install_ascend_qsa_cache_split(qsa_module=qsa_module)
        install_ascend_qsa_multihead_store(
            qsa_impl_cls=qsa_module.AscendQSAImpl,
            qsa_module=qsa_module,
        )
        implementation = object.__new__(qsa_module.AscendQSAImpl)
        implementation.head_size = head_size
        implementation.num_kv_heads = kv_heads
        implementation.do_kv_cache_update(
            None,
            key,
            value,
            (key_cache, value_cache),
            slot_mapping,
        )
        torch.npu.synchronize()

        actual_key_cache = key_cache.cpu()
        actual_value_cache = value_cache.cpu()
        expected_key_cache = torch.zeros_like(actual_key_cache)
        expected_value_cache = torch.zeros_like(actual_value_cache)
        key_cpu = key.cpu()
        value_cpu = value.cpu()
        for row, slot in enumerate(slot_mapping_cpu.tolist()):
            page, offset = divmod(slot, page_size)
            expected_key_cache[page, offset].copy_(key_cpu[row])
            expected_value_cache[page, offset].copy_(value_cpu[row])
        if not torch.equal(actual_key_cache, expected_key_cache):
            raise RuntimeError("multi-head QSA key cache differs from exact store reference")
        if not torch.equal(actual_value_cache, expected_value_cache):
            raise RuntimeError("multi-head QSA value cache differs from exact store reference")

        logical_indices_cpu = logical_positions.reshape(2, 6).to(torch.int32)
        logical_indices = logical_indices_cpu.to(device=device)
        block_table = block_table_cpu.to(device=device)
        token_to_req = torch.zeros(2, dtype=torch.int32, device=device)
        query = torch.randn(
            (2, query_heads, head_size),
            dtype=torch.bfloat16,
            device=device,
        )
        actual = triton_attention(
            query,
            key_cache,
            value_cache,
            logical_indices,
            block_table,
            token_to_req,
        )
        expected = reference_attention(
            query,
            key_cache,
            value_cache,
            logical_indices,
            block_table,
            token_to_req,
        )
        torch.npu.synchronize()
        relative_l2 = _relative_l2(actual, expected)
        max_abs = float((actual.float() - expected.float()).abs().max().item())
        if not bool(torch.isfinite(actual).all().item()):
            raise RuntimeError("multi-page QSA attention produced a non-finite value")
        if relative_l2 > 0.01:
            raise RuntimeError(
                f"multi-page QSA attention relative L2 exceeds 0.01: {relative_l2}"
            )

        payload = {
            "schema": 1,
            "status": "PASS_QSA_MULTIPAGE_MULTIHEAD_DEVICE",
            "started_at": started,
            "ended_at": time.time(),
            "device": str(device),
            "cache_shape": list(key_cache.shape),
            "row_shape": list(key.shape),
            "query_shape": list(query.shape),
            "block_table": block_table_cpu.tolist(),
            "logical_positions": logical_positions.tolist(),
            "physical_slots": slot_mapping_cpu.tolist(),
            "pages_touched": sorted(set(physical_pages.tolist())),
            "kv_heads": kv_heads,
            "query_heads": query_heads,
            "store_api": ASCEND_QSA_MULTIHEAD_STORE_API,
            "store_exact": True,
            "attention_relative_l2": relative_l2,
            "attention_max_abs": max_abs,
            "key_cache_sha256": _tensor_sha256(actual_key_cache),
            "value_cache_sha256": _tensor_sha256(actual_value_cache),
            "attention_sha256": _tensor_sha256(actual),
            "reference_sha256": _tensor_sha256(expected),
        }
        _write(args.output, payload)
        print(payload["status"], json.dumps(payload, sort_keys=True), flush=True)
        return 0
    except BaseException as error:
        payload = {
            "schema": 1,
            "status": "FAIL_QSA_MULTIPAGE_MULTIHEAD_DEVICE",
            "started_at": started,
            "ended_at": time.time(),
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        _write(args.output, payload)
        print(payload["status"], json.dumps(payload, sort_keys=True), flush=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
