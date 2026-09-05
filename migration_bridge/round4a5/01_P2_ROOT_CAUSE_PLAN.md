# Round 4A.5 P2 根因定位与修复方案

目标：在不改变 P2 placement、numerical contract、CPU hit coverage 或系统语义的
前提下，定位并修复 P2 same-path nondeterminism。

## 1. 已知边界

必须保留的反证：

- P1 单层 4 CPU Expert 确定；
- P2 Q2/H2 numerical contract 通过且 overflow=0；
- P2 4/4 layers、16/16 CPU experts coverage 通过；
- overlap 和历史 sequential control 均失败；
- 新 disposable container 已复现；
- Graph、Deferred Experts 和 Dynamic Placement 均关闭。

因此当前不能通过更换 experts、降低 CPU hit、放宽 `B_pair`、删除 prompt 或全局
同步来关闭问题。

## 2. 当前优先假设

### H1：多个 layer wrapper 共享 CPUInfer/task queue

相同 worker 配置会从 `_cpu_infer_instances` 取得同一个 CPUInfer。P1 只有一个
CPU-enabled wrapper；P2 有四个 wrapper。需要确认跨层 submit/sync 是否消费了错误
generation 的任务或完成状态。

### H2：SGLang 跨层共享 staging buffer

KTEP 的 `SharedStagingBuffer` 被多个 MoE layer 共用。即使 Python 调用顺序串行，
torch_npu/CANN copy completion 仍可能晚于下一层对同一 storage 的写入。

### H3：CPU/NPU partial 或 merge buffer 写覆盖不完整

Round 2B retained pipeline 曾出现约 `1.7e38` 输出。需要排除未初始化、残留、部分
写入或错误 dtype/byte-count。

### H4：特定新增 P2 layer 的局部实现问题

Layer 17 已由 P1 验证；Layer 1、9、26 可能触发不同 shape、mapping、weight offset
或 grouped-MoE 行为。

### H5：更底层同路径 kernel nondeterminism

只有在输入、route、weights、buffer generation 和 completion order 全部相同，而
单独 CPU 或 NPU partial 仍变化时，才进入该假设。

## 3. 执行顺序

### Step 1：matched-history 双运行捕获

固定同一 teacher-forced token history，先使用：

```text
v_en_01 token index 1
v_struct_01 token index 3
```

每次运行记录各层：

- layer input；
- router IDs / weights；
- CPU input / output；
- NPU premerge partial；
- merged routed output；
- shared expert output；
- layer output；
- final logits。

每个 tensor 记录 shape、dtype、device、data pointer、generation ID、finite、SHA256。
退出条件是找到第一个 run-to-run hash 不同的 stage。

### Step 2：diagnostic layer bisection

依次测试：

```text
{17}
{1}
{9}
{26}
{17,1}
{17,9}
{17,26}
{1,9,17,26}
```

每个组合至少 10-repeat。子集只用于归因，不能替代最终 P2 placement。

### Step 3：共享资源隔离 A/B

按以下顺序增加 default-off diagnostic switch：

1. 每个 layer wrapper 独立 CPUInfer；
2. 每个 layer 独立 staging buffer；
3. 每个 forward 独立 CPU input/output buffer；
4. submit 后记录 queue sequence，sync 后断言对应 generation 完成；
5. output 分配后 poison，在合法写入点清零；
6. merge 前检查 poison absence 和完整 write coverage。

一次只改变一个变量。任何恢复确定性的开关都必须回到 matched-history capture 中
验证第一个不同 stage 已消失。

### Step 4：局部同步证明 ordering hypothesis

只在已怀疑的边界加入 debug-only hard sync：

- staging copy -> D2H consumer；
- CPUInfer submit -> queue visibility；
- CPUInfer completion -> H2D；
- H2D completion -> merge；
- 当前 layer completion -> 下一 CPU-enabled layer staging reuse。

同步只用于定位。最终修复必须表达正确的 ownership/event/generation contract，不能
保留无范围的全局 synchronize。

### Step 5：最小 production fix

root cause 需要同时具备：

1. failure 前向证据；
2. 单变量 A/B；
3. 最小复现测试；
4. 不改变 numerical arithmetic 和 placement；
5. 去掉 diagnostic hard sync 后仍稳定。

## 4. 修复后的验收顺序

1. 首个 failing token matched-history regression；
2. 最小 failing layer set 10-repeat；
3. 原 P2 四 prompt 10-repeat；
4. 8/16/32/64 prefix exact；
5. CPU-not-hit exact；
6. P2 Q2/H2 frozen contract；
7. 4/4 layers、16/16 experts coverage；
8. P2 quality smoke；
9. Round 2A/2B/2C；
10. Round 3、P1 和 SGLang KT EP 回归。

全部通过后才允许构建 P3 GGUF并进入 P3。

## 5. 建议的首个实现任务

首先添加不改变 arithmetic 的 debug instrumentation：

- wrapper ID、layer ID、CPUInfer address；
- monotonic forward generation；
- task submit/start/finish/sync sequence；
- staging/CPU output/NPU partial 的 pointer 与 hash；
- matched-history capture runner。

首轮不要修改 buffer ownership。先取得第一个不同 stage，再决定优先实现
per-wrapper CPUInfer 或 per-layer staging buffer A/B。

