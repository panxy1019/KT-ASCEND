# Round6.4 执行基线与批准记录

最终更新2026-09-16：本轮执行已完成，结论`IMPLEMENTED_NO_SPEEDUP`，
不是加速READY。详见`ROUND6_4_FINAL_SUMMARY.md`与`20_FINAL_ADR.md`。
以下为历史开工记录，原始来源和审批边界保留不变。

日期：2026-09-15。状态：EXECUTION_STARTED，非资格通过声明。

历史开工记录：下文原大样本统计协议已被用户批准的
`14_MINIMUM_COST_QUALIFICATION.md`替代；不要按此历史记录启动旧矩阵。
已执行的小样本与正式性能结果分别见15、17；当前状态以执行检查点为准。

用户批准原文：`方案已阅，同意按此计划执行。质量容忍度 2pp 获批`。
获批方案为同目录 `ROUND6_4_OFFICIAL_DEFERRED_APPROXIMATE_ACCELERATION_PLAN.md`；
保留原稿 PLAN_FOR_REVIEW 作为审批前版本，本记录明确其现已获批。

## 来源

- 本地起点：`ba3494d9aedcc0e5623157832134e1e32a8ea44a`。
- 执行分支：`codex/round6-4-official-deferred-approximation`。
- SGLang：`f02f304a25364d1e8fe9c52515a22f20f5089a52`。
- Round5 reference：`round5-upstream-aligned-tp1-a3-verified` →
  `be5b48621db32fb894402e3e886060dea263cdc9`（本次重新解析）。
- 本地已有 `third_party/llama.cpp` 修改属于用户；本轮不提交或回退它。
- A3 `kt-r6-deferred-ascend` 为既有 disposable 容器；开工时只有 sleep infinity。
- A3 checkout + overlay 必须逐文件取证，不能用本地 HEAD 冒充实际加载版本。

## 批准的解释

2pp = 每个主 benchmark 的 ON 减 OFF 准确率差下限为 −0.02，
不是相对下降 2%，也不是两个 benchmark 平均后允许损失 2pp。
执行获批方案中的配对非劣效检验：每个 endpoint 单侧 97.5% 下置信界
严格高于 −0.02，两个 endpoint 均成立才通过。
区间跨界时 INCONCLUSIVE，不以“不显著”冒充非劣效。

承认跨层 Deferred 为近似；旧数值排序门禁不再作为本轮质量入口。
Round6.3 的 REJECTED 不修改。系统 exact-once、固定日程确定性、finite、
buffer/stream 生命周期仍是硬门禁。Deferred 默认 OFF。

## 执行检查点

1. WP0：审批、来源、运行协议与 quality 数据/统计冻结。
2. WP1：基础回归与非 Deferred compact reference；D1 系统检查。
3. WP2：S6 156 请求、独立 restart、oracle、1000 实际 expert forwards。
4. WP3：独立 batch2/4 工程验证；不得替代 batch1 主质量 endpoint。
5. WP4–5：诊断与独立质量；不再等待旧 QD3 candidate。
6. WP6–8：真实性能、稳定性、分离实现/质量/加速最终结论。

当前仅批准及分支建立完成；任何 A3 PASS 均须链接新原始结果。
主质量测试禁止启动，直到数据 revision、样本、模板、解析器、统计方法实现
及容量设置完整冻结；2pp 本身已批准，不再请求重复批准。

## 环境纪律

仅 NPU0、CPU0–15、本项目容器；不修改宿主库，不停止其他任务。
旧 512 context launcher 不足以支持正式 GSM8K 输入加 512 输出；必须进行
同配置 OFF/ON 的显式容量验证后再运行主质量集。
