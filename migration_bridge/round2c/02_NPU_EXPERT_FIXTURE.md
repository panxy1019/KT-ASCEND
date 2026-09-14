# Deterministic NPU Expert Fixture

## Scope

The NPU side is intentionally a single-layer deterministic fixture, not a model loader or production fused kernel. It reuses the Round 2A deterministic four-expert F32 weight generator and moves reordered gate/up/down tensors to NPU BF16 once during construction.

Implementation: `TorchNPUExpertProvider`, `kt-kernel/python/hybrid_moe.py:114-199`.

## Operator semantics

For each selected NPU-owned global expert:

```text
gate = linear(x, W_gate)
up   = linear(x, W_up)
h    = silu(gate) * up
y_e  = linear(h, W_down)
y   += routing_weight * y_e
```

This is ordinary public PyTorch/torch_npu execution (`F.linear`, `F.silu`, `index_add_`). It introduces no custom Ascend kernel, graph capture, HCCL, TP, quantization, full-model loading, private torch_npu ABI, or SGLang modification.

## Ownership and zero behavior

- Resident NPU tensors are indexed by **global logical ID** after one `argsort(physical_to_logical_map)` reorder (`hybrid_moe.py:151-160`).
- Only `placement.accelerator_experts` are executed (`hybrid_moe.py:187-198`).
- CPU-owned route weights are zeroed before accumulation. A CPU-only routing matrix therefore produces an exact all-zero NPU tensor.
- Output is BF16 on the same NPU as the input and retains the original input shape.

## A3 evidence

`test_ascend_hybrid_npu_expert.py:26-53` compares each resident NPU expert against an independent float reference. Lines 56-65 prove exact zero for CPU-owned routes; lines 68-83 prove physical-to-logical weight reorder.

The placement plus NPU fixture launch completed with **24 passed in 7.64s**. The final combined matrix also included all four NPU fixture tests in both successful 43-test runs.
