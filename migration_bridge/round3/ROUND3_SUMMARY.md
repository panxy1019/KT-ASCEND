# B Round 3 Summary

## Repository

KTransformers Round 2C/frozen Round 3 base:
`5b382086940f3c5f92bf52120fc1004e6d53026b`.

Branch: `feature/kt-deepseek-v2-lite-ascend-tp1`.

SGLang repository: `kvcache-ai/sglang`; base `0f36b26d`; final
`06b319dc3a62b77c880e36b042d273bfc3957d12`; branch
`feature/kt-ep-ascend-tp1`. Child patches are preserved under
`round3/sglang_patches/`.

## A3

- Kernel: openEuler aarch64 `6.6.0-72.0.0.76.oe2403sp1.aarch64`
- CPU visibility: 640 CPUs, 8 NUMA nodes; disposable container restricted to CPUs 0-15
- NPU: Ascend 910; `npu-smi 26.0.rc1`; only NPU 0 assigned to the test container
- Python: 3.11.15
- PyTorch: 2.9.0+cpu with torch_npu 2.9.0.post2
- Container: `kt-r3-dsv2lite`, 128 GiB limit
- Host/business containers: not modified

## Model

- DeepSeek-V2-Lite snapshot revision `604d5664`
- Config SHA256: `f346286b0f1c8b044252fd54cb4fa78b9fab6472a6e8bebb9edfe03d414ea03d`
- Tokenizer SHA256: `41f3bf64213da8c012d8bd0871a58a1fdf70463e8f08f110ddbb1082f529f669`
- CPU GGUF: layer 17, all 64 experts, F32; SHA256 `a16a50827ec81b54195bf246c7f9d05f7c1d5f3601ee33426c732f65892e180f`
- Final placement: layer 17 expert 8 only on CPU; SHA256 `05bae81924d79677c2ea03cbce4b74b6fa6e95e144389c6cdd4890fc4ad30f53`

## All-NPU Baseline

TP1 BF16, graph off, load/prefill/decode/generation all pass. NPU memory after
load was about 29.48 GiB. This is the full-model A/B source of truth.

## KT EP CUDA Dependency Audit

Hot-path stream/event/native-handle assumptions were accelerator-enabled.
Import-time CUDA probes and router CUDA JIT were guarded. Advanced CUDA-only
FP8/MXFP host-registration and weight-streaming paths are explicitly outside the
BF16 MVP. No mechanical whole-file CUDA-to-NPU rewrite was performed.

## SGLang KT Ascend Bridge

The bridge uses the current native ACL stream, an auxiliary accelerator stream,
events, and Round 2B CPUInfer. A decode race in torch_npu D2H staging was found
and fixed with exact `acl.rt.synchronize_stream()` plus synchronous ACL D2H/H2D
and owned host tensor lifetime. CPU and NPU expert compute still overlap.

Synthetic E2E: 4 SGLang tests passed. Round 2C final: 44 passed. Independent
`qlen=1` cold-process stress: 10/10 passed.

## Real Weight Identity

Layer 17 expert 8: 10 CPU repeats were byte-identical; CPU vs BF16-rounded FP32
relative L2 `5.29e-5`; CPU vs NPU relative L2 `0.004299`; all finite.

## Real Single-Layer Hybrid

One CPU expert and 63 NPU experts. Router/global IDs, routing weights,
`routed_scaling_factor`, and shared expert each have one owner. CPU-not-hit,
mixed-hit, sequential, and overlap cases pass.

## Prefill, Decode, and Generation

The 15-case matrix covers three prompt families and 1/8/16/32/64 token budgets.
All Hybrid token IDs exactly match all-NPU. Expert 8 is proven hit three times;
every recorded layer-17 pass contains exactly 63 NPU experts and CPU ID 8.

Maximum observed logprob absolute difference was `0.08105409145` without token
divergence. Hybrid latency after warmup was generally 1.58-1.89× all-NPU.

## Stability

Two full campaigns each generated 576 tokens (3 cycles × 3 prompts × 64). All
requests exactly matched their baseline and all logprobs were finite. Final NPU
process memory was about 30,584 MiB and container memory about 55.83 GiB. No
ERROR, traceback, crash, deadlock, or monotonic leak was observed. The server was
stopped cleanly and NPU 0 was released.

## Regression

- Round 2A: 21 passed; 1,000-cycle RSS delta 0
- Round 2B: 23 passed, 1 expected skip; RSS delta 6,184,960 bytes
- Round 2C: 44 passed; RSS delta 6,184,960 bytes; overlap lower bound 1.156 ms
- SGLang KT EP: 4 passed

## P0

- TP: 1
- Graph: OFF
- Deferred: OFF
- Dynamic placement: OFF
- MTP: OFF
- Speculative: OFF
- Old integrated Ascend operator copy: NO

## Sub-Gates

```text
DEEPSEEK_V2_LITE_ALL_NPU = PASS
SGLANG_KT_ASCEND_BRIDGE = PASS
REAL_EXPERT_IDENTITY = PASS
DEEPSEEK_V2_LITE_SINGLE_LAYER = PASS
DEEPSEEK_V2_LITE_PREFILL = PASS
DEEPSEEK_V2_LITE_DECODE = PASS
```

## Final Gate

```text
DEEPSEEK_V2_LITE_TP1 = A3_VERIFIED_READY
```

This status is bounded to the configuration above. Graph mode, deferred experts,
dynamic placement, TP>1, MTP, speculative decoding, advanced quantized weight
streaming, and performance optimization remain future work rather than hidden
claims of this round.

