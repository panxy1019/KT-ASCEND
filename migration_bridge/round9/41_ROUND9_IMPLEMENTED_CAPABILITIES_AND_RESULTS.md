# KT-ASCEND Round9 已实现功能与实际效果

更新日期：2026-09-21

## 1. 功能总览

本轮把最初的 C128、低比特和 Dynamic 设计推进到了五层可执行能力：配置与身份、CPU/native 算子、设备专家迁移、SGLang 请求边界、完整模型资源门禁。各层都有明确输入、拒绝条件和证据，不依赖删除 guard 或环境变量绕过来“开启”功能。

```text
候选 profile 与来源身份
          ↓
四 NUMA / C128 / 160 宽切片
          ↓
BF16、Q8_0、MXFP4 完整 CPU 专家银行
          ↓
原始 BF16 → NPU 的有界 Dynamic 提升
          ↓
请求完成/中止清理后的 generation 提交
          ↓
完整模型 firstHTTP / system8 / canary / 4096 / cancel-reuse
```

## 2. 候选 profile 和 fail-closed capability

候选 profile 将以下维度分别描述：

- `workers`：64 或 128。
- `storage`：BF16、Q8_0、MXFP4。
- `placement`：static 或 request_boundary。
- `numa_nodes`：实际准入后选定的 NUMA 节点集合。

当前完整模型矩阵固定使用 `workers=128`、NUMA `[2,3,6,7]`。配置加载不仅检查 JSON 值，还检查 native library 是否导出对应 capability、候选 overlay 是否安装调度器边界、银行 receipt 是否匹配 storage/profile、Dynamic 设备 gate 是否匹配源码和银行 SHA。

实际效果是：

- 缺少 profile 时拒绝启动。
- C128 profile 不会静默回到 C64。
- Q8_0/MXFP4 不会静默回到 BF16。
- request-boundary Dynamic 不会静默变成 static。
- 未安装调度器边界或设备门禁不合格时，启动前失败。
- 每个完整模型试验冻结实际 import tree 和 payload SHA，后续文件变化不能冒充原试验身份。

## 3. C128 四池拓扑

C128 使用四个 NUMA 子池，每个子池 32 个 worker。640 宽 CPU expert 中间维度被切分为：

- `[0:160]`
- `[160:320]`
- `[320:480]`
- `[480:640]`

每个子池计算 F32 partial，确定性归并后才进行最终 BF16 cast。这样避免每个分区过早 BF16 舍入造成的累积误差。

准入同时检查：

- 128 个不同物理 core identity。
- 不把同一个物理核的 SMT sibling 重复计数。
- 另外保留 8 个辅助 CPU，总 quota 为 136。
- worker 线程 CPU affinity 和 NUMA memory affinity。
- 完整模型运行时实际存在 128 个单 CPU 绑定线程，并且 128 个 CPU 值互不重复。

实际效果：六个最终完整模型组合都记录了 `128 pinned threads / 128 distinct pinned CPUs`。

## 4. 低比特权重格式和完整银行

### 4.1 格式

Round9 的 CPU routed expert 支持：

- Q8_0：32 weight/block，34 bytes/block，GGUF tensor type 8。
- MXFP4：32 weight/block，17 bytes/block，GGUF tensor type 39。

converter 明确定义了 scale、舍入、packing、非有限值拒绝和 underflow 处理。scalar codec 保留为默认参考，向量 encoder 通过了 1025 block byte-identical 检查，生成的 tiny GGUF 文件 SHA 与 scalar 路径一致。

### 4.2 银行完整性

每种 storage 均有 48 层完整 CPU backing。银行验证包括：

- 每个 GGUF 文件的完整字节 SHA-256。
- tensor 类型和 shape。
- 512 logical experts 的完整覆盖。
- layer0 与 layer1-47 的目录和文件集合。
- 原始模型 receipt、144 个 live file size、config/index SHA。
- 只读挂载文件在启动时的 size、mtime、inode 和内容签名。

实际效果：完整模型加载时不能只凭 catalog 文件存在就接受银行，也不能把其他来源、其他 storage 或不完整 sample bank 接到候选 profile 上。

### 4.3 低比特 native 数值效果

修复后的隔离 native buildv2 在 C64/C128 上通过 Q8_0 和 MXFP4 的多 token 长度与重复测试。

- 跨池 F32 partial 对照最大 relative L2：Q8_0 `1.24856e-7`，MXFP4 `1.06529e-7`，门限 `1e-5`。
- 相对“量化权重 + Q8_0 激活”参考：Q8_0 最大误差约 `0.000256949`，MXFP4 最大误差约 `0.000566466`。

这些结果证明 native 分区、尾部和归并实现与本轮量化参考一致。它们不证明相对原始 BF16 模型的最终精度等价。

## 5. 原始 BF16 提升与批量 NZ 写入

Dynamic 提升不从低比特 CPU bank 解码，而是从原始 BF16 checkpoint 的指定 expert/projection 有界读取。来源 reader 校验结构和运行时文件签名，避免在执行中读取被替换的源文件。

设备写入采用 projection 级批量 staging：

- 依次读取需要变更的原始 BF16 experts。
- 将同一 projection 的数据组成一个 staging buffer。
- 通过父张量 `index_copy_` 一次写入实际 NZ 权重。
- 所有 projection 同步完成后才发布新的 logical/physical map 和 ownership mask。
- 最大显式设备 staging tensor 为 400 MiB；这是 tensor 上限，不是 allocator 或 workspace 峰值。

设备门禁 v15/v16/v17 分别覆盖 BF16、Q8_0、MXFP4，均通过：

- 64 resident experts 反转并恢复，内容完全一致。
- promote → demote → promote 三代变更。
- 9 个 KTEP case，覆盖 q=1、10、32 和 mixed/resident-only 路径，每个重复 3 次。
- actual processed tensor format 为 NZ 29。
- native ownership mask 指针在原位更新期间保持稳定。

## 6. Dynamic generation 状态机

Dynamic 控制器为每个请求创建 lease，并绑定进入请求时的 generation。请求路由观察必须携带匹配的 request ID 和 generation；过期 lease、重复 ID、活动请求期间计划、缺失 layer 或不完整计划都会拒绝。

请求结束流程为：

1. 完成推理输出或收到 abort。
2. 清理 request pool、mamba pool、KV/GDN/PLE 消费者。
3. 标记 `consumers_drained=True`。
4. 根据已收集路由生成确定性的 64 experts/layer 新计划。
5. 复制新进入的 experts，保留仍驻留 expert 的 slot。
6. 设备同步。
7. 原位更新 map/mask，并在锁内发布新 generation。

任何 copy、sync 或 publication 异常都会将状态置为 `BROKEN_RELOAD_REQUIRED`。代码没有提供“只回滚 metadata 后继续服务”的伪恢复路径，因为被覆盖的设备权重未必已经恢复。

实际完整模型事务结果：

| Storage | Dynamic 事务 | Final generation | Expert copies | 已清理的必需测试请求 |
|---|---:|---:|---:|---:|
| MXFP4 | 18 | 18 | 22,072 | 16/16 |
| Q8_0 | 18 | 18 | 21,740 | 16/16 |
| BF16 | 18 | 18 | 21,440 | 16/16 |

18 个事务包括 2 个启动 health 请求以及 firstHTTP、8 个 system、4 个 canary、context4096、active abort 和 post-abort reuse 共 16 个测试请求。generation 均为连续的 1 至 18。

## 7. SGLang 请求边界接入

候选 overlay 在真实 SGLang scheduler 正常完成和 abort 路径中安装 pre-forward lease 与 post-release commit hook。完整模型验证不是用测试 double 代替服务器，而是通过实际 HTTP 请求驱动该边界。

实现保证：

- 一个请求的 chunk 期间保持同一 generation。
- request pool index 和 mamba pool index 清理后才允许迁移。
- abort 事件具有真实 `finish_reason.type=abort`。
- active abort 后可以执行新请求，并完成对应 Dynamic 事务。
- static 路径不建立 Dynamic 事务，且要求成对输出 token trajectory 严格一致。

Dynamic 的 post-abort reuse 只证明服务继续产生有限输出、状态已清理且边界事务完成；它不声称输出 token 必须与 static baseline 一致。

## 8. 独占资源准入和监控

候选完整模型采用隔离资源模式：

- 只映射 `/dev/davinci15` 和必要管理设备。
- 容器 network 为 none。
- 运行前检查物理卡 15 无其他 PID。
- 检查 4 个 NUMA 节点各自 free memory 至少 80 GiB。
- 检查 host MemAvailable 至少 704 GiB。
- 检查 CPU idle、worker topology、模型/银行来源和所有预算 receipt。
- container 设 136 CPU quota、640 GiB RAM，无额外 swap。

监控器只绑定实际 Docker container ID 的 cgroup，并持续检查：

- owned process HBM。
- board HBM。
- 外部未知 NPU PID。
- host cgroup memory 和 fail count。
- 容器存活状态。

超过 guard、发现未知 PID 或遥测失败时，监控器只停止本轮 owned container，不操作其他容器。受控停止之前的峰值才会写入 trial summary，退出过程中错误读到 root cgroup 的样本不能污染推理峰值。

## 9. 完整模型五项驱动

每个最终组合都使用同一类五项验证：

1. `PASS_FIRST_REAL_HTTP_PREFILL_DECODE`：首次真实 HTTP prefill/decode。
2. `PASS_FULL_MODEL_SYSTEM8`：四个 prompt 各执行两次，共 8 次；static 验证严格重复，Dynamic 验证有限输出和边界 drain。
3. `PASS_FOUR_KNOWN_ANSWER_CANARIES`：391、北京、ORANGE、严格 JSON 四项。
4. `PASS_4096_TOKEN_RESOURCE_REQUEST`：实际 3840 prompt tokens 和 256 completion tokens。
5. `PASS_ACTIVE_ABORT_AND_STATE_REUSE`：真实 streaming abort、清理和后续复用请求。

结果封存将 `resource_system_status` 和 `content_canary_status` 分开，但 `status` 不会忽略内容失败：只要 canary 失败，候选整体仍失败，finalizer 和矩阵都返回非零。

## 10. 最终功能效果

### 10.1 已达到

- C128 四池 topology、native partition 和完整模型线程绑定可执行。
- Q8_0 与 MXFP4 的 48 层完整 CPU expert bank 可加载。
- 低比特 C128 native 算子通过量化参考数值门禁。
- 三种 storage 的真实 64 expert Dynamic 提升和回迁设备门禁通过。
- SGLang 正常完成和 abort 请求边界可驱动真实迁移。
- BF16、Q8_0、MXFP4 的 static/Dynamic 六个完整模型资源组合全部完成。
- 五个组合的资源、系统和严格四项内容检查通过。
- BF16 Dynamic 完成资源系统检查，但严格 JSON 格式失败被正确保留。
- 所有原始结果、运行身份、峰值和事务元数据有可独立重算的本地证据。

### 10.2 尚未达到

- 完整 256 题精度对比。
- 低比特相对 BF16 的模型级质量判定。
- Dynamic full-model same-placement numeric equivalence。
- candidate clean restart qualification。
- BF16 Dynamic strict JSON pass。
- 性能排序和生产部署资格。

## 11. 结果边界

本轮最重要的结果不是“所有模式已经可用于生产”，而是三条候选路径已经具备真实实现、真实拒绝条件和完整模型资源证据。后续质量、重启、数值 oracle 和性能工作可以建立在这套框架上，但必须分别取得新资格，不能从当前功能/资源 PASS 自动继承。
