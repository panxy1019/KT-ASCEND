# Routed MoE Correctness

## Status

`PASS` — 4/8 experts、top-2、decode、prefill-like、多种 routing 与 physical/logical mapping 均在 A3 production ARM extension 上通过。

## Coverage

| Dimension | Cases |
|---|---|
| experts | 4, 8 |
| top-k | 2 |
| qlen | 1, 2, 8, 32, 64 |
| mapping | identity `[0,1,2,3]`; permutation `[2,0,3,1]` |
| router weights | `0.7/0.3`; `1.0/0.0`; token-varying weights |
| routes | same experts across tokens; different experts across tokens |
| layers | 0 and 1 through lifecycle suite |
| token layout | flattened token semantics |

人工 route `[1,3]`、weights `[0.7,0.3]` 的 reference 为：

```text
y = 0.7 * E1(x) + 0.3 * E3(x)
```

独立 reference 证明 selected experts 各贡献一次、non-selected experts 不贡献、router weight 没有重复应用。`weight=0` 的分支允许 BF16/kernel 舍入级非零残差，按同一 numerical thresholds 判定，而不使用不合理的 bit-exact zero 断言。

## physical_to_logical_map invariant

API 契约是 `physical_id -> logical_id`：GGUF 中第 `physical_id` 个 weight slice 被放入 runtime 的 `logical_id` slot。

- Python validation：`kt-kernel/python/utils/llamafile.py:LlamafileMoEWrapper.load_weights:186-201`，要求 CPU、int32、长度精确且为完整 permutation。
- C++ application：`kt-kernel/operators/llamafile/moe.hpp:LLAMA_MOE_TP.load_weights:194-230`，以 `physical_to_logical_map[physical_id]` 选择目标 logical slot，并做 bounds check。

Permutation `[2,0,3,1]` 的 routed outputs 与按同一映射组织的独立 PyTorch reference 一致。这是后续 hybrid expert placement 必须保持的 invariant。

## Prefill hidden-size defect

- classification：`PREFILL`
- reproduction：早期最小 fixture 使用 `hidden_size=32`；qlen 1/2/8 走 `forward_one` 可完成，qlen 32/64 进入 grouped `forward_many` 后 output non-finite/未完整写入
- first failing stack：`LlamafileMoEWrapper.forward -> MOE.forward_task -> LLAMA_MOE_TP.forward_many` 的 down stage
- source：`kt-kernel/operators/llamafile/moe.hpp` grouped down work partition 使用 `hidden_size / QK_K`；`QK_K=256` 且 hidden 32 时 block count 为 0
- scope：LLAMAFILE grouped/prefill path 的隐藏维契约，不影响合法对齐模型
- fix：`kt-kernel/python/utils/llamafile.py:LlamafileMoEWrapper.__init__:89-102` 明确验证 hidden 和 intermediate 都可被 256 整除；fixture 默认 hidden 调整为最小合法 256

修复选择是把 backend 的真实约束在 Python 边界 fail-fast，而不是在本轮重写 C++ prefill tiling。

## Numerical results

BF16 I/O、F32 fixture weights、seed `20260827`。所有 case 的 worst observed：

```text
max_abs_error     6.103515625e-05   <= 1e-3
mean_abs_error    1.4901161193847656e-08 <= 1e-4
relative_l2_error 3.9346272514744805e-05 <= 1e-2
```

### Acceptance correction

- classification：`NUMERICAL`（test acceptance issue，不是 production defect）
- reproduction：`weight=0` case 曾要求 bit-exact zero，observed residual 为约 `3.8e-6` 至 `6.1e-5`
- first failing stack：pytest exact-zero assertion
- source：`kt-kernel/test/per_commit/test_llamafile_routed_correctness.py`
- scope：测试判据
- fix：统一使用明示的 BF16 max/mean/relative-L2 阈值，并继续打印每个 case 的三个指标

## Batch semantics

当前 wrapper 原生语义是任意 leading dimensions flatten 为 token count，再恢复原 shape；并非独立的 batch/sequence kernel ABI。qlen=2 已覆盖相当于 batch=2 的 flattened token 输入。文档明确这一点，不虚构 backend 不存在的 batch 维。
