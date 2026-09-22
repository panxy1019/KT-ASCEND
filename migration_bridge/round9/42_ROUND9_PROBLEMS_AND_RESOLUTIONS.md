# KT-ASCEND Round9 问题与修复复盘

更新日期：2026-09-21

本文逐项记录本轮 C128、低比特和 Dynamic 开发中实际暴露的问题。每项包含现象、根因、修复、验证和残留边界，避免后续只看到最终 PASS 而重复走弯路。

## 1. C128 不能用可见逻辑 CPU 数直接推导

### 现象

早期资源思路曾考虑把旧可见 CPU 集合直接扩大为 128 worker，但该集合中包含 SMT sibling；“有 128 个逻辑 CPU”并不等于“有 128 个物理 core”。

### 根因

原有 C64 资源模型只覆盖两个子池，对四池 C128 缺少物理 core identity、NUMA 分布和辅助 CPU 预算检查。若只按 CPU 编号计数，可能让两个线程落在同一物理核上，同时破坏 NUMA locality。

### 修复

- 新增 topology plan，输入每个 CPU 的 `socket:core` identity 和 NUMA node。
- C128 固定四个 32 worker 子池，每个子池一个 NUMA 节点。
- 过滤 SMT sibling，要求 128 个不同 core identity。
- 另外预留 8 个辅助 CPU，总 quota 为 136。
- 完整模型运行时读取 scheduler 所有线程的 `Cpus_allowed_list` 和 `Mems_allowed_list`，生成 affinity evidence。

### 验证

六个最终完整模型组合均观察到 128 个单 CPU 绑定线程和 128 个不同 CPU。NUMA 选择为 2、3、6、7，每池 32 worker。

### 残留边界

该 topology 绑定 110.123.0.3 当前硬件和容器准入；换主机或共享模式必须重新发现，不能复制旧 NUMA ID。

## 2. Q8_0 ARM tile 路径产生非有限值

### 现象

native low-bit tile v1、v2、v3 在 Q8_0 路径出现非有限输出。即使尝试紧密 packed input，问题仍存在。

### 根因

ARM IQK Q8 路径期望 `block_q8_0_x4` 激活组布局，而调用方 `from_float` 实际产生普通 Q8_0 block。权重类型相同掩盖了激活布局不匹配，最终 vec-dot 读取了错误组织的数据。

### 修复

不再让该调用形态进入 IQK x4 激活路径。Q8_0 权重改为调度到现有 tinyBLAS standard-layout 路径，并保留 capability gate，防止未来无条件选回不匹配实现。

### 验证

隔离 native v4 覆盖 24 个 shape，通过：

- padded `lda/ldb/ldc`。
- 160/320 reduction。
- 2560 gate input。
- 64 tail。
- token 数 1、10、32。

后续完整 Q8_0 C128 static 和 Dynamic 均通过资源/系统及四项严格内容检查。

### 残留边界

通过的是当前标准布局调用链；若未来显式生成 x4 激活布局，需要单独实现和验证，不能仅切换 dispatch 标志。

## 3. 160/320 分区触发低比特尾部限制

### 现象

旧 llamafile multi-pool path 只对 F32/BF16 宣称 640 宽尾部安全。低比特 `forward_many` 还包含 256 对齐条件，C64 的 320 和 C128 的 160 都可能落入未支持分支。

### 根因

旧实现把分区对齐条件和数据类型能力混在一起，默认了更宽的 tile 对齐。32-weight block 格式本身能表示 160/320，但 native loop 没有明确处理对应尾部。

### 修复

- 在 `tp_slices.hpp` 中增加 32 block 对齐的低比特切片能力。
- 在 `moe.hpp` 的 `forward_many` 路径中增加受 capability 保护的尾部处理。
- Python wrapper 只在 native 导出匹配能力、ARM DOTPROD 可用且当前构建没有错误 SVE 假设时允许该组合。
- 对 160 和 320 直接做 isolated native shape tests，而不是从 640 结果外推。

### 验证

Q8_0/MXFP4 在 C64/C128 下均覆盖 q=1、2、9、10、11、32 和三次重复；实际原始 10 expert CPU MoE 的 ownership flip 也通过。

### 残留边界

能力和当前 ARM build identity 绑定。重新编译不同 ISA 选项后需重做 native gate。

## 4. 低比特 FP32 TP partial 被遗漏

### 现象

早期 wide MoE 测试看似通过，但严格的 C64/C128 相同输入对照在 Q8_0 q=1 出现 relative L2 `0.0021650733`。

### 根因

Q8_0/MXFP4 没有进入 `fp32_tp_partial_` 路径。每个 pool 的 partial 在归并前被过早转换为 BF16，之前的参考测试也使用了相同的提前舍入，因此没有暴露错误。

### 修复

- 将低比特 storage 纳入受 capability 保护的 F32 TP partial 路径。
- 四池 partial 以 F32 归并，最终才 BF16 cast。
- 新增跨 C64/C128 相同权重、相同输入的严格回归，不再让参考实现复制同一错误。

### 验证

修复后的隔离 native buildv2：

- Q8_0 跨池最大 relative L2 `1.24856e-7`。
- MXFP4 跨池最大 relative L2 `1.06529e-7`。
- 门限 `1e-5`。

第一次新回归容器因缺少 `SYS_NICE` 在 NUMA bind 时退出 134；补齐 CPU-only 测试容器 capability 后，保留 RED/Green 两份证据。

### 残留边界

该修复证明分区归并一致性，不代表低比特模型相对原始 BF16 的质量不下降。

## 5. 设备 tensor format 被错误标为 NZ

### 现象

早期 v2-v5 日志和描述称迁移目标为 NZ，但真实环境中的 `format_cast` 返回的是 ND base format。

### 根因

测试只验证了内容和调用成功，没有读取 `torch_npu.get_npu_format` 的实际值。代码对“调用过 format_cast”与“最终 tensor format 是 NZ”做了错误等价。

### 修复

- v6 显式读取 w13/w2 的实际 NPU format，确认二者均为 ND 2。
- 修正文档：早期结果只证明 processed weight 内容和迁移/merge 正确，不授予 NZ 资格。
- adapter 在相应阶段显式拒绝非预期 format，后续再通过真正 NZ probe 取得资格。

### 验证

v6 在候选 binary 下完成 BF16+C128+Dynamic local merge，但被准确标为 ND qualification。随后 v15/v16/v17 在实际 NZ 29 下完成批量迁移和 KTEP 验证。

### 残留边界

历史 v2-v5 原始日志保留，不能改写为 NZ PASS；后续应以 v15-v17 为设备 format 资格来源。

## 6. 对 materialized slice 写入没有更新父 NZ tensor

### 现象

早期 actual-NZ expert probe 中，代码对 `destination[slot].copy_()` 写入后，父权重没有按预期更新，导致设备门禁不能证明真实 resident tensor 已替换。

### 根因

在 NZ 布局下，索引表达式可能产生 materialized tensor，而不是对父存储可原位写入的 view。对临时 tensor `copy_` 成功不等于父 tensor 内容已经改变。

### 修复

改为在父张量上使用 `index_copy_`，并在写入后重新读取目标 slot，验证 logical expert 的 w13/w2 内容、ownership 和 KTEP 输出。

### 验证

expert v12/v13/v14 的 single-expert actual-NZ 测试通过；最终 v15/v16/v17 扩展到 64 expert reverse/restore 和三代迁移。

### 残留边界

未来若更换 NPU format 或权重容器结构，必须重新验证“写父 tensor”语义，不能假设所有 indexing 都返回 view。

## 7. Dynamic 逐 expert NZ copy 太慢

### 现象

完整模型 v12 能完成 48 层、424 expert 的第一代迁移和第二次 8 expert 迁移，但第一次请求边界耗时超过 4 分钟。CPU quota 没有被 throttle，py-spy 和读计数显示进程仍在真实推进。

### 根因

每个 expert、每个 projection 都单独执行 NZ resident write，形成大量小 `index_copy_` 和同步操作。算法正确，但请求边界开销不适合作为完整模型功能路径。

### 修复

- 按 projection 批量组装原始 BF16 experts。
- w13 最大 staging 约 400 MiB，w2 约 200 MiB，顺序执行，避免同时持有多个大 staging buffer。
- 每个 projection 只执行一次父 tensor `index_copy_`。
- staging 释放前完成设备同步。
- 全部 48 层复制完成后才全局发布 mapping；中途失败进入 `BROKEN_RELOAD_REQUIRED`。

### 验证

BF16/Q8_0/MXFP4 的 64 expert reverse/restore gate 全部通过。三个最终 Dynamic 完整模型组合各完成 18 个请求边界事务，无边界超时。

### 残留边界

400 MiB 是显式 staging tensor 上限，不是完整 allocator/workspace 峰值。完整模型 HBM 仍以 monitor 实测为准。

## 8. 完整模型 allocator 上限继承了共享模式参数

### 现象

full-model v13 完成首次 HTTP 和两个真实 Dynamic commit 后，申请 402 MiB 时触发 PyTorch allocator OOM。日志显示 38.38 GiB active、39.13 GiB reserved、39.50 GiB allowed，但设备仍有 21.72 GiB free；Docker `OOMKilled=false`。

### 根因

候选 SGLang overlay 复制了共享卡基线中的 40,448 MiB allocator cap。用户已把物理卡 15 改为独占，但候选构造器仍沿用旧共享预算，因此是进程 allocator policy OOM，不是板卡物理容量耗尽。

### 修复

- 只在隔离 candidate overlay 中把 allocator limit 调整为 43,008 MiB（42 GiB）。
- 要求显式 `exclusive` resource mode 和新鲜 server admission receipt；共享模式不能使用更高上限。
- owned guard 调整为 45,056 MiB（44 GiB）。
- admission model budget 调整为 46,080 MiB（45 GiB），另保留 8 GiB 和实际 board baseline。
- 基线 source/runtime 和共享模式 41,472 MiB guard 保持不变。

### 验证

22 个资源预算回归通过，包括“共享模式必须拒绝 exclusive 上限”。最终三个 Dynamic 组合 owned HBM 峰值为 41,060、41,060 和 41,062 MiB，低于 44 GiB guard。

### 残留边界

这些预算只适用于独占模式。若重新共享物理卡 15，必须重新测量训练峰值并使用共享预算或重新设计共存上限。

## 9. 监控器在容器退出时可能读错 cgroup

### 现象

早期完整模型停止过程中，Docker top/container 状态切换产生竞态；旧监控器可能在容器 cgroup 消失后读到 root cgroup，并把巨大的宿主内存值记入 trial peak，或把自然退出误报为 guard failure。

### 根因

监控目标通过易变化的容器查询结果解析，没有始终绑定创建时的 Docker ID 和确定的 cgroup path。受控停止之后的退出样本也没有与推理期峰值严格分开。

### 修复

- 创建容器后冻结完整 Docker ID。
- 优先解析 Docker ID 对应 cgroup v1 path，并兼容混合 cgroup 主机。
- inspect/cgroup 解析失败时 fail closed，但只停止 owned container。
- finalizer 以 `controlled-stop.json.time` 为边界，只计算停止之前的资源峰值。
- 显式区分 `OWNED_EXITED_DURING_SAMPLE` 和 `STOP_OWNED`。

### 验证

最终 v19-v23 monitor 中没有 `STOP_OWNED`；server log 均记录 SIGTERM、drain、remaining 0 和 graceful exit。独立审阅重新按 controlled-stop 时间计算峰值，与 summary 完全一致。

### 残留边界

v10 的旧错误样本仍保留为历史证据，并另存停止前正确 summary。不能删除历史日志或把退出竞态说成推理期内存峰值。

## 10. 候选身份和银行绑定不够严格

### 现象

独立审阅发现早期候选可以在缺少显式 profile 时走 fallback，Dynamic 可以被环境变量单独打开，profile 中的 storage 也没有强制绑定到实际加载的 48 层 GGUF bank。

### 根因

配置意图、安装的候选代码、native library、银行内容和设备 gate 分散在不同位置，启动器没有构成完整闭环。仅凭目录名或 catalog 很容易把错误银行、旧 overlay 或部分 sample 接入完整模型。

### 修复

- 候选模式要求显式 profile；缺少 profile 不允许 fallback。
- Dynamic 必须同时满足 profile、scheduler hook、adapter module 和匹配 device gate。
- 48 层全部绑定同一 storage/profile/source identity。
- verifier 重算全部 bank 文件内容 SHA，并检查 GGUF type/shape 和文件集合。
- overlay receipt 同时绑定 preserved baseline tree 和 candidate tree。
- launcher/server 重新计算实际 imports、payload、profile、bank receipt、source receipt 和 native runtime SHA。
- 每个 trial 将实际 payload 和 admission receipt 冻结进证据目录。

### 验证

最终六个 trial 的 summary 分别绑定 13 或 14 个证据文件；独立审阅重新计算所有哈希，无缺失或不匹配。bank receipt 和 Dynamic device gate SHA 均能在独立证据目录找到对应原件。

### 残留边界

身份绑定证明“运行了哪套代码和数据”，不证明代码本身正确；因此仍需算子、设备、内容和资源门禁。

## 11. NUMA free memory 门禁被候选银行文件缓存阻塞

### 现象

v19 完成后，下一项 v20 准入拒绝：NUMA node 2 只有 69.13 GiB free，node 7 只有 77.20 GiB，低于每节点 80 GiB 门限。此时没有 Round9 作业运行。

### 根因

只读排查显示主要占用是刚使用过的候选 Q8_0/MXFP4 GGUF 文件页缓存，并非匿名内存泄漏或外部训练进程。全局 drop cache 会影响无关容器，降低门限又会削弱门禁。

### 修复

- 保持 80 GiB 门限不变。
- 仅在对应 candidate 已停止后，对本轮拥有的 96 个低比特 GGUF 文件调用 POSIX `FADV_DONTNEED`。
- 不修改文件内容、不全局 drop cache、不操作原始 checkpoint 和无关容器。
- 每个低比特 trial 停止并封存后，只对本 trial 的 bank 做定向 cache advice，再重新采样准入。

### 验证

四个 NUMA 节点恢复到 80 GiB 以上，v20-v23 均通过新的独立准入。before/after 和文件列表记录在 `20260920-owned-bank-cache-reclaim-v1.json` 及各 trial 的 `post-stop-owned-cache-advice.json`。

### 残留边界

如果未来原始 checkpoint cache 成为主要占用，必须先确认是否有其他活跃消费者；不能把本次“owned generated bank”策略扩大成任意文件缓存清理。

## 12. 严格 canary 在解析失败前丢失原始回答

### 现象

BF16 Dynamic v18 的前三项 canary 通过，第四项在 `json.loads` 处失败。旧 driver 在 append row 之前解析，因此异常发生后第四项 raw response 没有被写入证据。v18 也因此在 context4096 和 cancel/reuse 之前停止。

### 根因

证据捕获顺序错误：将内容解析成功作为保存 response 的前提。这样在最需要原始回答诊断时反而丢失数据。

### 修复

- HTTP response 返回后立即保存完整 request、response 和 output IDs。
- 在 JSON 解析之前计算并记录 `numeric_finite`。
- 解析或内容不匹配时，在同一 row 保存 `passed=false` 和 `parse_error`。
- 不去除 Markdown fence，不修改 prompt，不更改 expected answer。

### 验证

v23 重现第四项失败并完整记录原始文本：正确对象被包在 Markdown `json` code fence 中。前三项严格通过，第四项 `numeric_finite=true`、`passed=false`、`JSONDecodeError`。

### 残留边界

证据修复解决了“看不见回答”的问题，没有解决 BF16 Dynamic 为什么选择 fenced 输出。

## 13. 内容失败和资源失败曾被同一 fail-fast 逻辑混在一起

### 现象

旧矩阵在任意 canary 失败后立即停止，导致 v18 无法继续证明 context4096 和 cancel/reuse 的资源/系统行为。另一方面，如果简单允许 canary 失败后继续，又可能错误地把最终候选标为 PASS。

### 根因

finalizer 只有一个聚合结果，没有区分：

- 内容不符合预期，但 HTTP 成功、响应非空、所有数值有限。
- HTTP/解析流程故障、响应为空、非有限值、资源 guard 或其他技术失败。

### 修复

- 新增 `resource_system_status` 和 `content_canary_status` 两个轴。
- 仅当 canary 的每一 row 都有有限数值、非空 raw response，且失败是保存过的内容不匹配或对应 parse error 时，才允许继续独立资源驱动。
- 非有限值、HTTP 错误、证据缺失或其他技术失败仍立即停止。
- 即使资源轴通过，只要内容失败，overall 仍为 `FAIL_CANDIDATE_CONTENT_CANARY`，finalizer 和最终矩阵均返回非零。
- 新增分类回归，验证技术失败不能被错误降级为内容失败。

### 验证

v23 在严格 JSON 失败后继续完成实际 4096 token 和 abort/reuse；最终准确记录：

- `resource_system_status=PASS_CANDIDATE_RESOURCE_SYSTEM`
- `content_canary_status=FAIL`
- `status=FAIL_CANDIDATE_CONTENT_CANARY`
- matrix `COMPLETED_WITH_CANDIDATE_FAILURES`
- matrix exit code 1

### 残留边界

双轴报告用于保留更多真实信息，不是放宽候选准入。任何对外汇总必须同时显示整体失败。

## 14. BF16 Dynamic 输出 JSON code fence

### 现象

最终 v23 第四项要求仅输出 JSON，实际返回：

````text
```json
{
  "name": "apple",
  "count": 3
}
```
````

对象字段和值符合语义目标，但第一个字符是反引号，严格 `json.loads` 在 column 1 失败。

### 当前判断

这是一项可重复观察到的严格内容格式失败。现有证据不足以判断它由以下哪一因素造成：

- BF16 Dynamic 迁移带来的细微数值轨迹变化。
- 请求历史和 resident placement 不同。
- 模型本身在无结构化输出约束时的格式选择。

系统 8 请求中的 fenced JSON 不能替代 canary 证据；v23 已提供同一 canary 的完整 raw response。

### 已采取措施

- 保留严格 parser 和原 prompt。
- 保存完整输出、token IDs 和 logprobs。
- 保留 overall FAIL，不把“语义正确”改写成严格 PASS。
- 完成资源轴，确认没有非有限值、HTTP 失败、迁移未 drain 或资源 guard 触发。

### 验证

v23 其他四个驱动通过；18 个 Dynamic generation 连续，16 个测试 request ID 全部 drain；owned HBM 峰值 41,062 MiB。独立复核确认 summary 的 14 个证据 SHA 全部匹配。

### 未解决项

这是本轮明确保留的开放问题。下一步需要 fresh-static same-placement numeric oracle：相同初始权重、相同 resident placement、相同请求历史，对比关键层输出和最终 token trajectory。若之后引入 grammar/structured output，应作为新的产品能力验证，不能用于改写 v23 历史结果。

## 15. 证据可搬运性不足

### 现象

独立审阅指出原 `post-trial-audit.json` 没有记录 `trial-summary.json` 自身 SHA，也没有记录生成 audit 的程序 SHA；summary 因生成更早，也没有反向绑定 post-trial audit。

### 根因

post-trial audit 最初被设计为同目录内的第二次检查，而不是可单独搬运的最终 receipt。

### 修复

没有改写已经封存的原始 trial 文件。新增矩阵级 `delivery-index.json`，绑定：

- 六份 trial summary SHA。
- 各 post-trial audit SHA。
- audit program SHA。
- matrix driver、matrix status 和 final device state SHA。
- 独立审阅 receipt SHA。

### 验证

本地重新计算所有 summary 中列出的 13/14 个原始证据文件 SHA，全部匹配。独立审阅重新计算 static repeat、Dynamic generation、copy 计数、请求 ID、资源峰值和最终设备状态，无 P1/P2 问题。

### 残留边界

新增索引提高了整包证据的自描述性，但原始 post-trial audit 格式未做历史重写。未来新 trial 可以直接在 audit 中加入 summary SHA 和 audit tool SHA。

## 16. 本轮问题处理原则

这些修复共同遵循五条原则：

1. 不删除 guard 来制造支持；新增 capability 并让未支持路径显式失败。
2. 不用 host、小样本或单层结果替代完整模型资格。
3. 不重写失败历史；修复后使用新版本、新目录和新 receipt 重跑。
4. 不把资源通过、内容通过、质量通过和性能通过合并为一个模糊 PASS。
5. 不为获得绿色结果而放宽 prompt、parser、资源阈值或身份绑定。

最终结果因此保留了一个真实失败：BF16 Dynamic 的严格 JSON canary。这个失败不会阻止我们确认它的资源系统轴，但也不会被资源 PASS 覆盖。
