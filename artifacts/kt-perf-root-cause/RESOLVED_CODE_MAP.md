# KT W2 性能根因收口：实际代码与依赖图

本文记录 v506、v507、v509 实际加载的冻结路径。行号针对远端冻结快照，不代表当前公开主线。Git HEAD 没有被当作运行身份；生产 native 与诊断 native 分别登记。

## 1. CPU Expert 的提交、NPU 并行和合并

实际文件：

`/mnt/data1/kt-round9/kt-root-cause-full-host-a-v509/overlay-python/sglang/srt/layers/moe/kt_ep_wrapper.py`

关键范围：

- `KTEPWrapperMethod._submit_cpu_forward()`：5541–5549，向 native wrapper 提交 CPU 工作。
- `KTEPWrapperMethod._sync_cpu_forward()`：5551–5560，取当前 device stream handle 后调用 native `sync_forward()`。
- staged input submit/sync：5618–5658。
- `KTEPWrapperMethod.apply()`：5660 起，是完整 Hybrid MoE 父调用。
- staging copy 与 CPU submit：5928–5955。
- logical expert 到 resident slot 的 mask/remap：5957–5968。
- resident NPU expert：5974–6031。
- CPU result sync、ready event、同层 merge：6033–6070。

冻结实现已经具有同层 CPU/NPU overlap：CPU 在 staging copy 后提交，resident NPU expert 随后在主 stream 运行，最后才取 CPU 结果并合并。诊断没有删除同步、改变精度、改变路由或提前发布部分结果。

## 2. Native TaskQueue 直接事件

实际文件：

`/mnt/data1/kt-round9/round11-t2-attribution-native-src-v3/kt-kernel/cpu_backend/task_queue.cpp`

关键范围：

- trace clock 与 1 MiB buffered JSONL：22–62。
- enqueue admission：88–104。
- `TaskQueue::sync()` 的 begin/end 与真实 `pending`：106–128。
- worker start、task body、complete 和 pending decrement：130–166。

事件只在 `KT_ROOT_TASK_TRACE_PATH` 非空时开启。补丁不改变 queue predicate、任务体、worker 数、NUMA、pending 次序或同步语义。

生产 native SHA-256：

`614ae60f70ff70d1e47d7fd32e857f887f874b6248eb332172c8a637cb471a6d`

诊断 native SHA-256：

`bccee617542c1a23ed1531e1099d1bff6190f8212c91f806b9ea705bec2b8a62`

## 3. 全层 Host 父调用观察

实际文件：

`/mnt/data1/kt-round9/kt-root-cause-full-host-a-v509/overlay-python/sglang/srt/layers/round9/round16_a1_observer.py`

关键范围：

- 全 48 层 Host 覆盖和三层 device sampling：30–31。
- `time.monotonic_ns()` Host 父调用：104–149。
- Routed Expert parent：210–242。
- 48 层 `_submit_cpu_forward`、`_sync_cpu_forward` 和 parent 绑定：245–289。
- scheduler batch 生命周期：294 起。

Host 记录只做时钟读取和内存中的 row append。Device event 延迟到既有 batch 完成边界查询，没有新增逐算子 `synchronize()`。v509 相对 v506 只将 GDN/QSA Host 父调用覆盖从抽样层扩展到全部 36/12 层；device event 仍只采早、中、晚层。

## 4. GDN、QSA、PLE 与剩余主干

实际模型文件：

`/mnt/data1/kt-round9/kt-root-cause-full-host-a-v509/overlay-python/sglang/srt/models/qwen4_exp.py`

关键范围：

- PLE prepare/run 与 state copy：520–558。
- HyperConnection、PLE 注入、attention 前后 mix/combine 和 MLP：561–601。
- 36 个 GDN layer 的 `linear_attn` 调用：604–622。
- 12 个 QSA layer：625 起。
- QSA qkv、Q/K norm、RoPE、KV state、indexer 和 sparse attention：643–690 及其后续 projection。

因此 `TOKEN_ACCOUNTING` 中的 `other_dense_dispatch` 不是一个未知的 100 ms callback。它绑定到以下实际区域：HyperConnection mix/combine、attention/MLP 外层、router/dispatch/combine、shared expert、norm/residual、sampler、scheduler gap，以及诊断观察开销。它是代码范围明确但没有继续拆成每个小算子的聚合项。

`DEPENDENCY_GAP_ACCOUNTING` 又使用同一 scheduler monotonic clock 上的父调用边界，将该聚合项精确分到 attention→同层 Expert、Expert→下一层 attention、请求首尾和 PLE 两侧。该重建没有新增设备运行，也没有把异步 gap 误写成纯 Host CPU 时间。

## 5. 本轮分析代码

本地文件：

- `tools/kt_perf_closure/build_native_task_trace_source.py`：构建 env-gated TaskQueue trace source。
- `tools/kt_perf_closure/build_profile_snapshot.py`：冻结 v506/v507 诊断快照与 source gate。
- `tools/kt_perf_closure/build_full_host_snapshot.py`：从 v506 派生 v509，只扩大 Host 父调用覆盖。
- `tools/kt_perf_closure/analyze_profile.py`：用同一 monotonic clock 连接 submit、task、sync、merge 和 scheduler batch。
- `tools/kt_perf_closure/summarize_regions.py`：排除嵌套 QSA indexer 后汇总 GDN/QSA/PLE/device samples。
- `tools/kt_perf_closure/build_token_accounting.py`：形成 125 个稳态 token 的全层顶层父调用分区和 CSV。
- `tools/kt_perf_closure/build_gap_accounting.py`：用同钟相邻依赖边界细分 residual，并拒绝重叠父区间。

对应回归位于 `tests/kt_perf_closure/`。所有新增行为均先有失败测试，再实现最小代码。

## 6. 实际依赖关系

```text
hidden/router ready
  ├─ staging copy → CPU submit → C128/MXFP4 task → CPU result ready
  └─ resident Q32 BF16/NZ29 NPU expert → NPU result ready
                                  ↓
                    existing stream/TaskQueue boundary
                                  ↓
                       exact same-layer merge
                                  ↓
                         next layer may consume
```

18,000 个被直接连接的层调用中，CPU result 在进入 TaskQueue wait 前均已 ready。当前 slow path 的依赖方向因此是 NPU/Host 前段决定何时到达 merge 边界，CPU worker 没有在 merge 边界形成可见尾部。
