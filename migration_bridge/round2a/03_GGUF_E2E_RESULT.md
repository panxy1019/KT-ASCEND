# GGUF End-to-End Result

## Status

`PASS` — A3 上完成以下真实 production 路径：

```text
generated local GGUF
  -> GGUFLoader
  -> KTMoEWrapper dispatch
  -> LlamafileMoEWrapper.load_weights()
  -> MOEConfig / MOE.load_weights_task
  -> CPUInfer.submit + sync
  -> LlamafileMoEWrapper.forward()
  -> LLAMAFILE MOE output
```

测试没有绕过 wrapper 直接构造 C++ MOE。production CPU forward 在 `kt-kernel/python/utils/llamafile.py:LlamafileMoEWrapper.forward:255-295` 中把输入按 flattened-token 语义组织为 BF16 hidden、int64 expert IDs、F32 router weights，再通过 CPUInfer submit/sync 执行。

## Assertions

- fixture 可重复生成且字节相同；
- loader 找到 gate/up/down 三个 production key；
- wrapper 能从 GGUF load；
- output shape 与输入一致；
- output dtype 为 BF16；
- output 全部 finite；
- repeated invocation deterministic；
- 与独立 PyTorch F32 reference 在 BF16 接受阈值内一致。

## Numerical contract

| Item | Value |
|---|---|
| input/output dtype | BF16 |
| weight / GGUF type | F32 |
| seed | 20260827 |
| hidden/intermediate | 256 / 256 |
| experts/top-k | 4 or 8 / 2 |
| max absolute threshold | `1e-3` |
| mean absolute threshold | `1e-4` |
| relative L2 threshold | `1e-2` |

整轮 routed matrix 的 worst observed：

```text
max_abs_error     = 6.103515625e-05
mean_abs_error    = 1.4901161193847656e-08
relative_l2_error = 3.9346272514744805e-05
```

误差明显小于阈值。阈值考虑 BF16 output rounding；权重为 F32，因此这里没有量化误差。

## Production fixes revealed by E2E

### CPU-only pinned allocator

- classification：`OTHER`（CPU-only allocation capability）
- reproduction：在 image 的 CPU-only PyTorch 下构造 all-CPU expert mask；`pin_memory=True` 报没有 pinned allocator
- first failing stack：`KTMoEWrapper -> BaseMoEWrapper` 的 expert mask allocation
- source：`kt-kernel/python/experts_base.py:_allocate_cpu_expert_mask:21-29`
- scope：只影响无 accelerator pinned allocator 的 CPU-only runtime
- fix：保留 pinned-memory 优先路径，仅在明确的 pinned allocator RuntimeError 时 fallback 到普通 CPU bool tensor；其他 RuntimeError 继续抛出

这不是 NPU/pinned transfer 路径实现，也没有改变存在 pinned allocator 时的行为。

## Evidence

- A3 focused E2E：`logs/round2a/gguf-e2e-rerun.log`
- final combined matrix：`logs/round2a/full-test-matrix.log`
- combined result：`21 passed in 4.82s`
