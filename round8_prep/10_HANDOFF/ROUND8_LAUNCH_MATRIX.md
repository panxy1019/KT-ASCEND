# Runtime launch matrix — gated templates, not executed

| Order | Profile | Slots/layer | Purpose | Entry |
|---|---|---:|---|---|
| 0 | Config / dense / GDN / one expert | n/a | Construction and primitive tests | Compatible isolated image, sampled weights |
| 1 | Q64-C64 | 64/256 | Static engineering quality reference | System/operator gates and memory admission |
| 2 | Q32-C64 | 32/256 | 12.5% heavy resident target | Q64 stable and frozen reference |
| 3 | Q16-C64 optional | 16/256 | Additional capacity point | Q32 qualified and actual capacity need |

All: TP1, max-running-requests1, greedy, context initially4096, chunked prefill512 (proposed, freeze after CLI capability check), BF16/unquantized NPU, text-only, shared fusionOFF, radixOFF, GraphOFF, DynamicOFF, DeferredOFF, MTP/OFF, speculativeOFF. No CUDA-only tutorial flags.

Command template (must substitute verified paths/backend and check actual `--help`; not an executable qualified launcher):

```bash
python -m sglang.launch_server \
  --model-path <PINNED_QWEN35_SNAPSHOT> \
  --kt-weight-path <VERIFIED_CPU_ARTIFACT> \
  --kt-method <ARM_VERIFIED_BF16_OR_LLAMAFILE> \
  --kt-cpuinfer 64 --kt-num-gpu-experts 64 \
  --tensor-parallel-size 1 --max-running-requests 1 \
  --context-length 4096 --chunked-prefill-size 512 \
  --disable-cuda-graph --disable-radix-cache \
  --disable-shared-experts-fusion --language-only \
  --host 127.0.0.1 --port 18400
```

Attention backend, threadpool count/NUMA mapping and explicit uniform mask-loader wiring require inspection of final runtime. Do not claim this template enforces all OFF features: record parsed server args/defaults and assert effective values before serving. Generated JSON masks are audit fixtures, not yet proven native loader input.

Full weights: use pinned `hf download Qwen/Qwen3.5-35B-A3B --revision 59d61f3ce65a6d9863b86d2e96597125219dc754 --local-dir <ISOLATED_MODEL_DIR>` only after disk/network/RAM admission and no timed Round7 contention. Verify every local shard against manifest before conversion. Current Hub LFS SHA is not local verification.

Ordered runtime gates: construction → dense/GDN → same-operand routed expert → shared expert → compact generation → Q64 determinism/restart/CMMLU → Q32 system and paired CMMLU. Q64 is a static engineering reference, not all-NPU correctness oracle. Observed quality delta≥−5pp, fixed256 examples and protocol; do not select masks by answer correctness. Freeze Qwen prompt/chat-template/token budgets and handling of thinking before first scored result.
