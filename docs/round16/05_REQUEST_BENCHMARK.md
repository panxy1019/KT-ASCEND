# 四进程公平请求基准

## 1. 比较性质

比较两套系统部署方案：

- **S0**：SGLang + KT；
- **B16**：vLLM-Ascend eager + KT Expert bridge。

模型 revision、tokenizer、prompt IDs、采样、输出预算、MXFP4 bank、Uniform Q32 map、C128 和物理12保持一致。框架和 graph 模式不同，所以结果只称系统比较，不称纯框架开销或纯 KT 消融。

## 2. 运行顺序

```text
S0-A -> B-A -> B-B -> S0-B
```

四个独立模型进程均 profiler OFF。每个进程执行：

- W1：128 输入／16 输出，1 warmup＋2 measurement；
- W2：128 输入／128 输出，1 warmup＋2 measurement；
- W4：64 输入／32 输出，1 warmup＋2 measurement；
- exact arithmetic canary 1 次。

每进程共 10 次 generation。所有 performance measurement 都生成完整 output budget，失败或短输出不会被静默删除。

## 3. 原始中位数与连续窗口吞吐

| Block | Workload | E2E s | TTFT s | TPOT s | serial output tok/s | requests/s |
|---|---|---:|---:|---:|---:|---:|
| S0-A | W1 | 6.28645 | 2.53259 | 0.25025 | 2.54516 | 0.15907 |
| B-A | W1 | 6.84086 | 2.47787 | 0.29087 | 2.33889 | 0.14618 |
| B-B | W1 | 6.66910 | 2.44062 | 0.28190 | 2.39912 | 0.14995 |
| S0-B | W1 | 6.42929 | 2.57318 | 0.25706 | 2.48861 | 0.15554 |
| S0-A | W2 | 34.69470 | 2.53940 | 0.25319 | 3.68932 | 0.02882 |
| B-A | W2 | 41.92371 | 2.53292 | 0.31016 | 3.05317 | 0.02385 |
| B-B | W2 | 42.35488 | 2.47587 | 0.31401 | 3.02208 | 0.02361 |
| S0-B | W2 | 36.01473 | 2.62679 | 0.26290 | 3.55410 | 0.02777 |
| S0-A | W4 | 9.23128 | 1.25805 | 0.25720 | 3.46647 | 0.10833 |
| B-A | W4 | 10.59951 | 1.36582 | 0.29786 | 3.01901 | 0.09434 |
| B-B | W4 | 10.96836 | 1.40353 | 0.30854 | 2.91748 | 0.09117 |
| S0-B | W4 | 9.22652 | 1.28832 | 0.25607 | 3.46826 | 0.10838 |

完整 CSV：[ROUND16_BENCHMARK.csv](../../artifacts/round16/ROUND16_BENCHMARK.csv)。

## 4. 配对比值

比值定义为 B16/S0；TPOT/E2E 小于 1 才是改善，throughput 大于 1 才是改善。

### Pair 1：S0-A / B-A

| Workload | E2E ratio | TTFT ratio | TPOT ratio | throughput ratio |
|---|---:|---:|---:|---:|
| W1 | 1.08819 | 0.97840 | 1.16231 | 0.91896 |
| W2 | 1.20836 | 0.99745 | 1.22502 | 0.82757 |
| W4 | 1.14822 | 1.08566 | 1.15811 | 0.87092 |

### Pair 2：S0-B / B-B

| Workload | E2E ratio | TTFT ratio | TPOT ratio | throughput ratio |
|---|---:|---:|---:|---:|
| W1 | 1.03730 | 0.94848 | 1.09661 | 0.96404 |
| W2 | 1.17604 | 0.94255 | 1.19442 | 0.85031 |
| W4 | 1.18879 | 1.08942 | 1.20493 | 0.84119 |

## 5. 晋级门

Decode 门要求两个 W2 配对块同时满足：

- TPOT ratio ≤ 0.90；
- serial output throughput ratio ≥ 1.10。

保护门要求 W1/W4 的 TTFT/E2E 不在任一配对块中超过 1.05。

实际结果：

- Decode gate：FAIL；
- protection gate：FAIL；
- disposition：`B16_MEASURED_NO_PROMOTION`。

W1 TTFT 在 B 中略有改善，但 W1 TPOT/吞吐、W2 Decode 和 W4 短请求均不支持切换默认承载。

## 6. 质量条件

每个正式块均验证：

- 固定 prompt IDs 和 output budget；
- 完整实际输出 token 数；
- finite logprobs；
- exact `391` canary；
- B 完整 smoke 另覆盖五类 canary、cancel 和 reuse。

因此性能差异不是由少生成 token 或过滤失败请求造成。

严格比较结果：[ROUND16_REQUEST_COMPARISON.json](../../artifacts/round16-b/ROUND16_REQUEST_COMPARISON.json)。
