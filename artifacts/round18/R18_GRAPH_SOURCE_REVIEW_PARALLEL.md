# Round18 冻结图模式与 Native-preserving seam 并行源码复核

日期：2026-09-30
性质：只读源码/证据复核；未启动 NPU、模型或容器；未修改运行代码。
审计目标：确认冻结 vLLM/vLLM-Ascend 对 `NONE`、`PIECEWISE`、breakable、`FULL_DECODE_ONLY` 的真实支持边界，解释 v011 的动态形状失败，核对 v013 完整 GDN opaque split 与上游模式的关系，并判断现有 CPU eager break 是否可以直接进入 FULL graph。

## 1. 结论

1. `NONE` 已在 Round18 当前计算身份下完成真实 48 层模型资格。`native-hybrid-full-smoke-v003` 使用 `enforce_eager=true`，最终解析为 `CompilationMode.NONE / CUDAGraphMode.NONE`，五类 canary、Decode32、cancel、reuse 均通过。这是当前唯一已经完成请求级正确性验证的 Round18 graph mode。
2. `PIECEWISE` 是冻结栈真实支持并实际解析到的模式，不是 CLI 假开关。v011 的 binding receipt 记录 `requested_graph_mode=PIECEWISE`、`resolved_graph_mode=PIECEWISE`、`resolved_compilation_mode=VLLM_COMPILE`，并记录完整 splitting-op 列表。v011 也完成 38 个 piecewise 子图的 AOT 编译；当前失败发生在首次 profile replay 的 Qwen GDN post-core range，而不是在 KT routed-expert 分流、权重创建或模型加载。
3. breakable cudagraph 在冻结 vLLM-Ascend 中被模块导入时硬设为关闭。`vllm_ascend/platform.py` 明确执行 `envs_vllm.VLLM_USE_BREAKABLE_CUDAGRAPH = False`，日志也明确说明 DeepSeek V4 PIECEWISE breakable 尚不支持。因此本轮不能通过环境变量“开启 breakable”，也不能把未运行写成未知性能。
4. 冻结栈具有一般性的 FULL/FULL_DECODE_ONLY 实现：full mode 会清空 splitting ops，并在 model runner 中用 `ACLGraphWrapper(... runtime_mode=FULL)` 包住完整模型；capture 时调用一次 Python/model forward，之后 replay 路径只调用 `entry.aclgraph.replay()` 并返回捕获输出。这个机制与当前 Round18 CPU bridge 不相容：当前 CPU submit/sync 位于 quant-method `apply()` 的 Python 路径内，没有 graph-external work scheduler、固定 CPU result buffer 更新协议或 replay 前 generation/event 注入。因此现有实现即使请求 `FULL_DECODE_ONLY`，也不能保证每个 Decode token 产生新 CPU work；真实资格为 `NOT_RUN_AND_CURRENT_SEAM_INCOMPATIBLE`。
5. v011 的 `1584 x 128` reshape 失败不是单页 KV shape 问题。实际张量有 `196608 = 32 * 48 * 128` 个元素，而 FX/AOT wrapper 使用 `1584 = 33 * 48` 作为首维。trace 同时显示 post-core range 接受形状为 `[s59,48,128]` 的 `core_attn_out`/`z` 和额外动态 scalar，生成的 wrapper 又从另一个 scalar 参数计算 `48*s26`。v011 将 `num_tokens` 改为 `z.size(0)` 后故障完全不变，说明错误位于 custom-op 分段后的动态 SymInt ABI/运行时绑定，而非某一行 Python 源码选错 tensor。该结论不包含任何“KV 已完全解决”的声明。
6. v013 的完整 Qwen GDN opaque split 在架构模式上与 vLLM v0.26.0 的 `olmo_hybrid_gdn_full_forward` 同构：两者都把完整 GDN forward 注册为 `vllm` custom op，以 caller-owned `output` 为 mutable 参数，通过 `get_forward_context().no_compile_layers[layer_name]` 找回真实层，fake 实现不推进状态，并将 op 列入 PIECEWISE splitting ops。v013 复用冻结 `AscendGatedDeltaNetAttention.forward`，不重写 GDN 数学/provider/state；但它是新本地 adapter，真实 PIECEWISE replay 和状态正确性仍需 v013 设备证据，不能由源码同构直接标 PASS。

## 2. 冻结身份和证据

### 2.1 冻结 vLLM-Ascend

本地冻结源根：

`/home/admin/Documents/ChatGPT/KT/kt-round9-mainline/artifacts/round15/upstream-image-source/round15-offline-vllm-source-20260928`

关键文件及 SHA-256：

| 文件 | SHA-256 | 用途 |
|---|---|---|
| `vllm_ascend/platform.py` | `c711042261d5283be5aa644c3a5a00fcb34f70c0156f4b866700f38a927a1276` | 图模式归一化、breakable 强制关闭、FULL/PIECEWISE 分支 |
| `vllm_ascend/ops/gdn.py` | `747933e091593c92731b3b00b5f336f5aa3270b702e011426d9a632954d41bcd` | 当前 Qwen GDN 实际 Ascend forward |

v011 binding receipt 还冻结了：

- Qwen GDN source：`1227d6f385a52296e9f08223544b1c5fdc7e8d9aa09a848e7a8e522a8dc51214`；
- Ascend MRoPE source：`6ce6f668ce3e10617fa8bbd9b7b6313a5a3fe16ea454f69a7c3d30432a85a04e`；
- v011 snapshot：`bdac2e06500850b503d3b510b901f44674d627930998eb3902542c12afbf003c`。

### 2.2 vLLM v0.26.0 上游参考

为核对 `olmo_hybrid_gdn_full_forward` 的具体模式，本次只读了官方 vLLM `v0.26.0` tag，对应 commit：

`568afb3a13806beb53bb2e6bd518269357b237c0`

关键上游文件：

- `vllm/model_executor/layers/mamba/gdn/olmo_gdn_linear_attn.py`，读取内容 SHA-256：`4e5b144d109c6f13431e1113ca8cf28132df14e7988dfaad08008cd1e43b18a1`；
- `vllm/config/compilation.py`，读取内容 SHA-256：`267f4ca04fdcd2f0cf55c334782dd498c8051b9500ad088673d79518fc712db0`。

上游参考只用于证明“完整 GDN custom-op split”是 vLLM 已采用的图边界模式；它不替代冻结 Ascend 栈的设备验证。

### 2.3 Round18 运行证据

| 证据 | 结论 |
|---|---|
| `artifacts/round18/evidence/native-hybrid-full-smoke-v003/result.json` | `PASS_ROUND18_NATIVE_HYBRID_EAGER_SMOKE` |
| `artifacts/round18/R18_HYBRID_EAGER_RESULT.json` | 48 层、512 logical/Q32 physical、`AscendMoERunner` 保留、graph claims 明确排除 |
| `artifacts/round18/evidence/native-hybrid-piecewise-diagnostic-v011/binding-receipt.json` | PIECEWISE 实际解析和 splitting ops；v011 shape adapter 身份 |
| `artifacts/round18/evidence/native-hybrid-piecewise-diagnostic-v011/container.log` | profile replay 的 `1584x128` vs `196608` 失败 |
| `artifacts/round18/evidence/native-hybrid-piecewise-diagnostic-v011/torch_trace/dedicated_log_torch_trace_pgd0uvir.log` | post-core 子图的 tensor shape、重复 scalar ABI 和生成 wrapper |
| `artifacts/round18/evidence/native-hybrid-piecewise-diagnostic-v012/container.log` | 仅测试工作目录错误；未启动模型，因此不是 v013 图方案的技术结果 |

## 3. 图模式支持矩阵

| Mode | 冻结栈源码支持 | Round18 当前实证 | 当前结论 |
|---|---|---|---|
| `NONE` | 支持；`cudagraph_mode=NONE` 会同时关闭 compile/Npugraph_ex/static/super kernel | v003 完整模型 smoke PASS | `QUALIFIED_CORRECTNESS`；性能 ABBA 尚未在本文内执行 |
| `PIECEWISE` | 支持；要求 `VLLM_COMPILE`，设置 splitting ops，禁用 Npugraph_ex/static/super kernel | v011 确认 resolved PIECEWISE 且 AOT 38 个子图；首次 profile replay 被 GDN SymInt ABI 阻塞 | `SUPPORTED_BUT_ROUND18_RUNTIME_NOT_YET_QUALIFIED` |
| breakable | 冻结平台在 import 时强制 `False` | 每次相关容器日志均打印 force disabled | `UNSUPPORTED_IN_FROZEN_STACK` |
| `FULL` / `FULL_DECODE_ONLY` | 一般性支持；full mode 清空 splitting ops，model runner 用 FULL ACLGraph wrapper | Round18 Native Hybrid 未运行；当前 CPU seam 无 replay generation 注入 | `FRAMEWORK_SUPPORTED_BUT_CURRENT_HYBRID_SEAM_INCOMPATIBLE` |

### 3.1 PIECEWISE 的实际含义

冻结 `platform.py` 对 `requires_piecewise_compilation()` 的路径执行：

1. 要求 `CompilationMode.VLLM_COMPILE`；
2. 通过 `set_splitting_ops_for_v1()` 建立 split；
3. 增加 Ascend 特定 attention op；
4. 明确关闭 `enable_npugraph_ex`、`enable_static_kernel`、`enable_super_kernel`。

因此本轮 PIECEWISE 是 FX/AOT 分段 + ACLGraph 的路径，不应称为 FULL/Npugraph_ex。v011 receipt 与初始化日志都与该源码一致。

### 3.2 FULL_DECODE_ONLY 的实际含义

冻结平台对 `has_full_cudagraphs()` 的路径将 `splitting_ops=[]`；model runner 随后将完整模型包装成 `ACLGraphWrapper(runtime_mode=FULL)`。`FULL_DECODE_ONLY` 是否对某次请求选择 FULL 由 runtime dispatcher 的 uniform-decode 判定决定，但一旦进入 FULL wrapper，其 capture/replay 行为与下面源码一致：

- capture：在 `torch.npu.graph(...)` 内调用一次 `self.runnable(*args, **kwargs)`；
- replay：只调用 `entry.aclgraph.replay()`，然后返回捕获图输出；
- replay 不重新执行 Python model forward。

所以“冻结框架认识 FULL_DECODE_ONLY”与“当前带 CPU expert 的 Hybrid 可以正确使用 FULL_DECODE_ONLY”是两个不同结论。前者为真，后者目前为假/未接通。

## 4. v011 duplicate SymInt ABI 根因

### 4.1 直接证据

v011 在同一进程中已完成：

- Native Hybrid live binding；
- `requested_graph_mode=resolved_graph_mode=PIECEWISE`；
- Qwen GDN/rotary source gates；
- compile range `[32,32]`、`[1,1]`、`[1,32]` 的 AOT 编译。

失败发生在 profile replay：

```text
_unsafe_view(clone, [mul_6, 128])
RuntimeError: shape '[1584, 128]' is invalid for input of size 196608
```

数值关系：

```text
1584   = 33 * 48
196608 = 32 * 48 * 128
```

trace 中 partition 前的 post-core GraphModule 明确显示：

```text
core_attn_out: bf16[s59, 48, 128]
z:             bf16[s59, 48, 128]
mul_504:       Sym(48*s59)
sym_size_int:  Sym(s59)
```

而生成的 AOT wrapper 又从单独 scalar 参数构造：

```text
mul_6 = arg6_1 * 48
_unsafe_view(clone, [mul_6, 128])
```

实际 tensor storage 属于 32-token profile range，但 `arg6_1` 在 replay 被绑定为 33。证据最符合“custom-op 分段后，等价 token 维被作为多个独立 SymInt/标量参数跨 ABI 传递，其中一个参数错误继承或错位绑定到相邻 range 值”的解释。

### 4.2 为什么 v011 的 `z.size(0)` 修订没有解决

v011 receipt 显示，源级修订将 token 数来源从早期 projection/hidden 改为 `z.size(0)`，同时声明不改变 math/provider/state。若问题只是 `num_tokens` 读取了错误 tensor，该修订应使 output allocation 与 `z` 一致。

但失败的 shape、目标元素数和 FX 行完全不变。trace 仍将 `z` tensor shape与 reshape scalar 作为独立 ABI 参数。由此可排除“仅换一行 token-size 来源即可修复”的假设；继续在原 forward 内做 `reshape`/`size` 微调没有新的根因依据。

### 4.3 明确边界

- 这不是 Q32 routed expert 物理容量错误；错误发生在 GDN post-core norm reshape。
- 这不是 CPU expert work-ID 或 completion 错误；失败发生在 profile 初始化阶段，还没有正式请求 replay。
- 这不是“单页 KV 加 `unsqueeze(0)`”问题；本次修订与证据没有扩大 KV shape 资格，所有 adapter receipt 继续写 `kv_contract_claimed=false`。
- 本文无法从 PyTorch 内部实现证明“具体哪一个调用栈赋值了 33”；因此把更细粒度的 PyTorch bug 定位标为 `UNKNOWN_INTERNAL_BINDING_SITE`，不把高置信的 ABI 现象冒充完整上游根因补丁。

## 5. v013 完整 GDN opaque split 与上游模式核对

### 5.1 v013 本地实现身份

| 文件 | SHA-256 |
|---|---|
| `round18/native_gdn_graph_boundary.py` | `74d80565a386a592246a84dd23f84f60b7e8b76d32a696f007571a726261b72e` |
| `round18_integration/native_hybrid_live_worker.py` | `aebf1726740f8ef118bfb9478a5808a44a9d62c46f7011aadbb7511146379419` |
| `artifacts/round18/run_native_hybrid_piecewise_diagnostic_v013_remote.sh` | `4d32763180878c93a4c0ef7ce0108f557f2f1bbe9f934e5b055498b997a1c3b1` |

本地 adapter 的关键结构：

1. 先校验冻结 `AscendGatedDeltaNetAttention` 文件 SHA-256；
2. 要求 Qwen GDN 当前 `forward` 就是冻结 Ascend forward；
3. 注册 `vllm::round18_qwen_gdn_full_forward(hidden_states, output, layer_name)`；
4. custom op 通过 `get_forward_context().no_compile_layers[layer_name]` 找到真实层；
5. 调用冻结 `original_forward(layer, hidden_states, output=output)`；
6. fake 实现不读取/写入递归状态；
7. 标记 `output` 为 mutable；
8. 把该 op 加入 PIECEWISE `splitting_ops`；
9. 仅用薄 wrapper 替换 Qwen GDN 的外层 forward。

### 5.2 与上游 `olmo_hybrid_gdn_full_forward` 的同构点

vLLM v0.26.0 的 Olmo GDN 采用：

```text
class.forward
  -> torch.ops.vllm.olmo_hybrid_gdn_full_forward(hidden, output, prefix)

custom op implementation
  -> get_forward_context().no_compile_layers[layer_name]
  -> self._full_forward(hidden_states, output)

registration
  -> mutates_args=["output"]
  -> fake_impl returns without state advance
  -> op name is in CompilationConfig._attention_ops / splitting ops
```

v013 与它在图边界、层查找、mutable output、fake 行为和 splitting-op 注册五个关键方面相同，因此可称为“架构模式同构”。

### 5.3 差异与未决项

| 项目 | 上游 Olmo | Round18 v013 |
|---|---|---|
| 实现归属 | 模型原生类内置 | 运行前、source-gated 本地 adapter |
| opaque body | `self._full_forward()` | 冻结 Ascend `original_forward()` |
| 上游动机 | 避免 GDN 递归状态放大 compile 数值差异 | 避免 Qwen GDN split 后重复 SymInt ABI |
| 输出参数 | caller 必须提供 | caller 未提供时 `empty_like`，随后仍传 caller-owned output 给 op |
| 资格状态 | vLLM 源码内模式 | 本地 mock 测试通过，真实 v013 replay 截至本文写入时尚无本地归档结果 |

所以 v013 是有明确上游先例的最小边界修订，不是随意把 GDN 改成 Python fallback；但只有真实运行能回答：

- custom op 是否成功成为 piecewise split；
- profile 和首次 replay 是否越过原 `1584x128` 故障；
- capture/warmup 是否未重复推进真实 GDN/QSA/PLE state；
- 每个 Decode token 的 CPU work ID 是否重新生成；
- GDN 完整 eager boundary 带来的图覆盖损失是否仍有性能价值。

本次复核对这些项保持 `NOT_RUN/UNKNOWN`。

## 6. CPU eager break 对 FULL 的决定性影响

当前 `NativeHybridQuantMethodAdapter.apply()` 在 Python 中执行：

1. 临时替换 module-global `select_experts`；
2. native selector 执行后调用 `cpu_submit(...)`；
3. 调用原生 Ascend quant method 得到 resident partial；
4. 调用 `cpu_sync(...)`；
5. 校验 request/layer/generation identity；
6. `native_result.routed_out += cpu_output`；
7. generation 计数递增。

这些动作都位于被 FULL `ACLGraphWrapper` 包裹的完整 model forward 内。冻结 FULL replay 只执行 NPU graph，不重新进入 `apply()`，因此当前实现缺少以下任何一种合法机制：

- replay 前的外部 CPU submit；
- 固定 result buffer 的 generation-scoped H2D/update；
- CPU ready event 与 NPU graph consumption 的正式依赖；
- stale result/cancel 的 graph-generation gate；
- capture/warmup CPU work 的隔离和 drain。

结论：当前 CPU eager break 仍然是 FULL/FULL_DECODE_ONLY 的决定性边界。FULL 不是“多加一个 splitting op”即可恢复，因为冻结平台在 full mode 主动清空 splitting ops。要让 FULL 正确，需要新的 graph-external CPU work orchestration，已经超出 v013 GDN shape 修订范围。

对 PIECEWISE 的判断更窄：理论上 custom-op split 可以让 Python/eager 边界每轮执行，但 v011 在 profile replay 早于正式请求时失败，尚未形成“每 token 每 layer CPU work ID +1”的真实证据。因此 PIECEWISE CPU generation 正确性必须由 v013 或后续成功运行记录证明，不能仅由代码结构推断 PASS。

## 7. 对主线的建议性结论

1. 继续 v013 PIECEWISE 是有依据的最后一次当前根因验证；它采用 vLLM 已有的完整 GDN opaque split 模式，而不是第三次 reshape 微调。
2. 若 v013 越过 profile/replay，应首先验证 resolved splitting op、GDN state/canary、CPU work ID 每 token/layer 更新，再做性能筛选；不能直接进入 FULL。
3. 若 v013 仍在相同 GDN/FX ABI 失败，当前冻结栈下的 Phase I graph recovery 可收口为 `PIECEWISE_GDN_BOUNDARY_BLOCKED`，不应继续叠加局部 shape patch。
4. breakable 在冻结栈中是源码级不可用，不建议升级整栈绕过本轮边界。
5. FULL/FULL_DECODE_ONLY 当前不能运行带 CPU complement 的正确 replay。若 PIECEWISE 能运行但不能超过 S0，这为进入 Phase II Native expert pool 提供直接架构依据：Decode 时去掉 CPU compute break，才有机会重新保留完整 Native FULL graph。

## 8. 未知项

- `UNKNOWN`：v013 在真实设备上的 compile、capture、replay 和完整模型结果；本文写入时本地尚无 `native-hybrid-piecewise-diagnostic-v013` 归档。
- `UNKNOWN`：v013 成功后 PIECEWISE 相对 Hybrid-NONE 的 TPOT/吞吐收益。
- `UNKNOWN`：冻结 FULL_DECODE_ONLY 在纯 Native Qwen3.8 TP1/当前 shape 上的独立性能；历史 TP8 Native 不能代替该数据。
- `UNKNOWN`：要保留 CPU complement 时，冻结栈是否存在未使用的正式 graph-external callback/event 接口。当前 Round18 adapter、ACLGraph wrapper 和现有 receipts 中没有这种接线。
- `UNKNOWN`：PyTorch/AOT 内部将 33 绑定到 reshape scalar 的精确内部函数；当前证据足以定位 ABI 边界，不足以声明上游 PyTorch 已知 bug 编号或通用修复。
