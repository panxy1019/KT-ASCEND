# Round6.4 官方与本地语义复核

Evidence：CODE_INSPECTED。本文件不是完整运行资格证明。

比较对象（本地 Git 对象存在，未 fetch/merge 动态 main）：

- 官方 KT `0dce4c9b097710951d1b3e232f50b973ff0a5505`。
- Round5 reference `be5b48621db32fb894402e3e886060dea263cdc9`。
- Round6.4 起点 `ba3494d9aedcc0e5623157832134e1e32a8ea44a`。
- child：Round5 `412219b4b05a888808f0b88d2963a8b6164a0fda` → 当前
  `f02f304a25364d1e8fe9c52515a22f20f5089a52`。

## 已检查的差异

| 位置 / 语义 | 核查 | 本轮动作 |
|---|---|---|
| experts_base.select_deferred_experts | TopK protected IDs 跨行 union；其余候选为 Deferred；相对 Round5 无新的 selector 实现替换 | 保留；batch2/4 实测 union 导致的曝光变化 |
| 两 slot、next-slot Deferred 输出、L+1 incremental merge | 已有官方结构，不是本轮新造 coordinator | 保留，补 oracle 与生命周期证据 |
| _clear_consumed_previous_pending | Round6 新增，consumer 提交后清除前层 pending，避免标记重复消费 | 保留，exact-once 硬门禁 |
| _has_cpu_routes | NPU/同步可见 pinned routes 上识别真正 CPU-owned work，抑制仅 NPU route 的空 Deferred task | 保留，whole-request no-hit 验证 |
| resolve_layer_max_deferred | child Round6 新增；紧邻 L+1 无 CPU wrapper 时 producer budget 归零 | 保留；D1 L8 terminal，原 sparse P1/P2/P3 不外溢 |
| output_dtype / buffer key | Round5 相对官方已携带 FP32 输出及按 dtype 区分 buffer；本轮未改 | 保留 FP32 merge，不退回提前 BF16 partial round |
| CPU standalone submit | Round5 让 CPU 无 device stream 时走 host submit/sync | 保留，属于旧 CPU regression 范围 |
| framework copy_ / 显式 NPU wait | 相对 Round5 未改执行 API | 保留，不恢复 raw ACL pointer copy |
| _round6_1_capture_tensor / child capture_tensor | stagewise/完整 forward 采集通过环境开关启用；OFF 时提前返回 | runtime OFF；诊断单独开启；计时不得带全采集 |
| KT_R61_EXPLICIT_REPLAY | 独立 pinned output，consumer 同步后 private contribution 合并 | 仅 oracle，不能作为生产 overlap 成绩 |
| KT_R61_CAPTURE_DRAIN_DEFERRED | 可以强制 drain 的诊断开关 | 正式 runtime/performance 不设 |
| ModelRunner apply_forced_history | 仅 KT_R63_FORCE_MANIFEST 存在时介入 next token | 正式自然生成不设；不修改默认 sampling |
| deepseek_v2 / KTEP stage hooks | 与 Round5 对比是采集点与 terminal guard，无新的 routing/shared/merge 架构 | 保留，需新 compact 与 D1 证明无回归 |

## 范围结论

Round5→当前 parent 的 kt-kernel 文件差异仅 experts_base.py 与新增 selection tests。
child 差异为 7 个文件，其中大部分是显式 opt-in 采集、teacher-force 和测试。
本轮不需要为了“接受近似”再搬一次完整 upstream 实现；先用已有代码关闭系统及质量证据。
这不是断言所有历史 patch 永久必要，也不是证明多 batch 的全局 buffer 自动安全。

正式 runtime driver 删除其子进程环境中的 KT_R61_/KT_R62_/KT_R63_/KT_DEBUG_ 开关，
保持 production overlap；route recorder 用于当前 compact，非性能计时。
原 BF16-visible expert boundaries 与 FP32 merge 不因取消全路径数值等价门禁而撤销。

## 尚需证明

独立四请求 oracle、全 S6 与 restart、真实 forward 生命周期、whole-request no-hit、
同 operands expert safety、batch2/4 buffer/RID/cohort，以及独立质量和速度仍须运行。
源码中存在某功能不等于 A3_VERIFIED_READY。
