# Deterministic Tiny GGUF MoE Fixture

## Fixture contract

生成器：`kt-kernel/test/fixtures/tiny_moe/create_tiny_moe_gguf_fixture.py`

默认配置：

| Field | Value |
|---|---:|
| seed | 20260827 |
| layers | 2 |
| experts | 8 |
| top-k | 2 |
| hidden size | 256 |
| intermediate size | 256 |
| weight dtype | float32 |
| GGML/quant type | F32 |

`hidden_size=256` 和 `intermediate_size=256` 是当前 LLAMAFILE decode/prefill 公共路径的最小安全 `QK_K=256` 对齐规模。它不是模型配置建议。

## Tensor layout and keys

每层包含：

```text
blk.N.ffn_gate_exps.weight  [experts, intermediate, hidden]
blk.N.ffn_up_exps.weight    [experts, intermediate, hidden]
blk.N.ffn_down_exps.weight  [experts, hidden, intermediate]
```

这些 key 完全遵守 production `GGUFLoader`/`LlamafileMoEWrapper.load_weights()` 契约，没有为测试增加别名或修改 production key。

权重用单次 `torch.manual_seed(20260827)` 后按 layer 顺序生成，缩放为 `randn * 0.02`，转为 contiguous F32。生成过程只依赖本地 Python、PyTorch、NumPy 和 gguf writer，不访问 Hugging Face、网络、模型服务或对象存储。

## Manifest

生成器在 GGUF 同目录写 `<name>.manifest.json`，记录：

- seed；
- layer/expert/top-k/shape；
- 每个 tensor 的 shape、dtype、GGML type、SHA-256；
- 整个 GGUF 文件 SHA-256。

## Reproducibility

A3 中独立生成 `tiny-moe-a.gguf` 与 `tiny-moe-b.gguf`，SHA-256 完全一致：

```text
e2a275952dd738223b45d05c220626b6103cffbfe732dc08537ccfa18b736247
```

byte-for-byte reproducibility test 也进入 pytest matrix。原始 fixture 和 manifest 位于远端 `logs/round2a/`，不是仓库中的大型二进制提交物；仓库提交的是确定性生成器与测试。

## Reference independence

测试 reference 从生成器返回的 canonical PyTorch weights 显式计算：gate projection、up projection、SiLU、element-wise product、down projection、router weighted merge。reference 不调用 `kt_kernel_ext`、MOE、CPUInfer 或 production LLAMAFILE kernel，因此不会与被测实现共享计算路径。
