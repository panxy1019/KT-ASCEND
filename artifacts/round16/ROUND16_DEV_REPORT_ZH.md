# Round16 DEV 阶段中文报告

日期：2026-09-28
终态：`ROUND16_DEV_READY_WAIT_ROUND15`

## 1. 本阶段实际完成了什么

Round16 按合同 60、61、62 完成了屏障前开发。A、B 两条承载路线分别位于独立稀疏 worktree，最终实现同步到 `kt-round16-integration`。本阶段没有调用 NPU、加载完整模型、读取完整 CPU bank、捕获图、运行 C128/NPU 性能基准、创建远端容器或修改 Round15 环境。

共享合同最终冻结为 `round16-contract-v2`。审阅发现 v1 的 frozen dataclass 仍可通过外部 list 引用改变 shape、route 或 writable-state 权限，因此 v2 增加深 tuple 冻结、整数/布尔边界、完整 completion 协议检查。v1 只保留为冻结前开发快照，未被静默覆盖。

## 2. A 线：保留 SGLang＋KT

A 线交付 `round16.a`，它是 provider 薄层，不复制 QSA、GDN、PLE 或 MoE 算法。SGLang scheduler、CPUInfer 和请求状态池仍是唯一所有者。provider 只完成以下工作：

- 把宿主 metadata 转换为不可变 `ProviderSpec` 与 `StateLease`；
- 在任何 backend、fallback 或 state write 之前核验 operation、phase、shape、dtype、layout、stride、device、stream、state format 和版本；
- 从实际 tensor 观察 shape/dtype/layout/stride/device，不能只相信 metadata；
- mapping 输入以 key 作为逻辑名，普通 torch/SGLang tensor 无需伪造 `.name`；序列输入按冻结 spec 的位置命名；
- 不支持的范围在状态写入前进入显式 fallback；声明与实际输入矛盾则直接 fail-closed；
- 在 backend/fallback 调用前冻结输入容器与全部顶层成员的身份快照，再校验 completion、lease、state-write 和输入/输出 alias，防止被调用方先删成员再返回该成员绕过 alias 门；
- close 对每个依赖执行 best-effort 回收，一个依赖失败不会跳过其他依赖，成功项不重复关闭，失败项可重试。

依赖按 P0/P1/P2 分类。P0 标准库合同已验证；P1 SGLang/KTransformers 接线使用显式注入；P2 torch_npu/CANN/真实算子仍是 `DEVICE_REQUIRED_NOT_RUN`。源码身份重新核对到 promoted Round14a v295，而不是旧 v235，六个小型源码文件的 SHA-256 均匹配。

A 离线结果：30/30 测试通过，状态为 `OFFLINE_COMPONENTS_READY_DEVICE_PENDING`。这表示可安装、可做宿主合同验证，不表示 Ascend 数值、完整层、完整模型、图或请求性能已通过。

## 3. B 线：vLLM-Ascend 承载 KT Expert

B 线交付 `round16.b`，范围是模型级 registry-shape scaffold、Hybrid Expert bridge、selective Q32 loader、生命周期适配和图断点合同。它没有替换全局 Ascend platform，也没有把 KV/QSA/GDN/PLE 所有权转交给 KT。

已闭环的离线合同包括：

- 原始 Top-10 只冻结一次，CPU/NPU 分支不重新 Top-K 或归一化；
- 全局 expert ID、Q32 physical slot 和 CPU bank logical ID 分开；
- 合并顺序固定为 `routed_weighted_sum * routed_scaling + shared_once`；NaN/Inf scaling 和 shared row 漂移被拒绝；
- loader 在 allocation 前验证完整 512 项 logical-to-physical map、恰好 Q32、连续 physical slots 和所有非 resident 的 `-1`；
- fused gate/up 和 tied alias 的所有权不能证明时，在零 allocation 状态 fail-closed；不会因名称覆盖而留下不可达 allocation；
- read/write 失败后逆序尝试释放全部唯一 allocation；cleanup 错误聚合为 cause，同时保留原始异常为主异常；
- request/allocation/batch/layer/token/completion/slot generation 共同决定 lease；非法 lease 不消耗 generation；
- cancel 后不接受新任务，late result 不能写入复用 slot；
- graph boundary 在 NPU/CPU 副作用前核验 active lease，每次 replay 要求非空输入指纹和单调 epoch；NPU replay 或 CPU submit 异常会退休预留 task；
- 两个不同输入的 replay 会产生不同 CPU identity 和结果，不能复用 capture 时的 CPU 输出。

真实 vLLM 0.26 model runner 构造、调用和 worker API 尚未在冻结设备镜像中解析。未提供 `VerifiedModelBinding` 时，注册返回 `VERSION_API_UNRESOLVED` 且不修改 registry。因此 B 当前是可安装、可测试的 scaffold，不是已经可用的 vLLM 模型插件。

B 离线结果：45/45 测试通过，状态为 `OFFLINE_BRIDGE_SCAFFOLD_READY_VERSION_DEVICE_PENDING`。真实 vLLM 模型、选择性 NPU 加载峰值、真实 MoE、decoder layer、完整模型、NPUGraph 和性能全部保持 `NOT_RUN`。

## 4. 集成验证

最终离线测试共 95 项：A 30、B 45、shared contract 14、Round15 barrier 6，全部通过。全部 Python 文件通过 `py_compile`；全部 JSON receipt 可解析。A、B 分别安装到新的临时目录：A 只包含 common＋A，B 只包含 common＋B；新鲜隔离进程导入时没有加载 torch、torch_npu、SGLang、KTransformers 或 vLLM，没有增加线程，也没有修改环境变量。

最终独立只读复审结论为 `NO_REMAINING_CODE_BLOCKERS`。复审重新确认了 95/95 测试、全部 Python 编译、逐文件空白检查，以及 A 的调用前身份快照、B 的 selective allocation/lifecycle/graph transaction 与 Round15 精确终态门。该结论只接受 DEV 代码冻结，设备项仍保持 `NOT_RUN`。

屏障逻辑使用精确终态 allow-list。`ROUND15_NOT_COMPLETE`、`ROUND15_INCOMPLETE`、`ROUND15_TERMINALIZING` 和带 `PENDING_CLEANUP` 的字符串均不能误开设备阶段。即使终态名称正确，baseline manifest、code/patch/dependency identity、cleanup PASS、空 owned process、lease released 和无 baseline correctness open issue 仍必须同时成立。

## 5. Round15 屏障与后续资源

最终只读核验显示 Round15 当前为 `P0_COMPLETE_A_COMPONENT_SCREEN_NEXT`，仍是执行中间态；没有终态 receipt，也没有 `resource_lease_released=true`。所以 Round16 后置设备阶段没有启动，当前真实终态只能是 `ROUND16_DEV_READY_WAIT_ROUND15`。

用户已授权在屏障开放后选择另一张安全 NPU 和兼容 NUMA。该授权不等于现在选卡，也不允许沿用瞬时空闲读数。VALIDATE 启动前必须先读取 S15 最后可信配置和更严格资源 cap，再 fresh 检查训练容器完整 ID/cgroup、所有 NPU 进程、未知进程、CPU/SMT sibling、NUMA 启动及运行余量、system memory、allocator、owned HBM 与 whole-card HBM。任何候选卡或 NUMA 不满足门槛即拒绝，不能自动退回物理 12 或历史 NUMA2/3/4/7。

## 6. 后置验证顺序和预算

屏障开放后只执行一次 S15 兼容整合：

1. 将 Round15 实际晋级内容组合成 S15；重解 A/B 的真实 symbol、ABI、metadata 与 state identity。
2. A：真实原语和权重 → 完整层 → 20 轮 shape/reuse/cancel → 模型 smoke → `S15/A16/A16/S15`。
3. B：真实 selective Q32 allocation → Hybrid MoE → decoder layer → 单 NPU eager 完整模型 → 合法分段图 → `A16/B16/B16/A16`。
4. W1=128/16、W2=128/128、W4=64/32；质量、状态、取消和资源门先于性能门。

后置硬上限仍为 14 个模型组、128 次 generation、6 次 abort、16 个设备组件进程和 32 NPU device-hours。本 DEV 阶段以上计数全部为 0，未借用 Round15 额度。

## 7. 当前不能写出的结论

- 不能写 A 或 B 已通过 NPU 数值、完整模型或请求性能。
- 不能写 vLLM 承载优于 SGLang，也不能写 B 已迁移成功。
- 不能把 A 对相同 compute 的封装说成新加速。
- 不能把 CPU/mock graph boundary 说成 NPUGraph 已通过。
- 不能把用户允许换卡解释成资源 admission 已通过。

本阶段的有效成果是两个隔离、可安装、可复验、fail-closed 的承载组件和一份不会提前打开设备阶段的屏障合同。
