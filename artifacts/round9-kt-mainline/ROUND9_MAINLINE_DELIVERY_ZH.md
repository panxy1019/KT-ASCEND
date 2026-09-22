# Round9 KT 主线核验与异构推理归因交付

日期：2026-09-22。执行范围：T0–T4、实验决策和 T6 收口。T5 PLE 行为实验未进入。

## 已完成什么

1. 冻结并核对原 v61 构建：203 个 source 文件和 2323 个 overlay 文件一致，核对模型、MXFP4 bank、native 扩展及运行依赖身份。模型完整文件校验采用已有可信全量 receipt 加当前大小/元数据检查；本轮没有重算整个 checkpoint。回放重新计算了 layer0 bank SHA。
2. 审计真实 GDN、QSA/indexer、Gated Residual、KTEP CPU/NPU 路径及实际 provider。官方资料仅作为边界参考，没有升级依赖。
3. 增加默认关闭的有界 provider 观测、真实 decode 捕获、保守时间轴汇总，以及可复用的真实专家回放入口。没有修改已冻结的 Layerwise、Dynamic、PLE 核心计算。
4. 完成两组观察关闭/开启的正常完整模型请求、一条真实 decode 捕获、中途 prefill 取消及同进程复用。两组可对照请求的输出 token 完全一致。
5. 在独立进程回放完整模型的 layer0 真实输入。C128 + MXFP4 CPU + Q32 BF16 NZ29 NPU 输出与捕获输出相同，连续三次哈希相同；CPU、NPU、合并结果分别通过混合精度参考校验。
6.49 项相关本地测试通过，冻结容器 host 检查通过，编译与源码差异格式检查通过（原始日志及逐字保留的任务书有4处空白提示，详见问题记录）。测试问题及审阅发现逐条记录在 [问题与修复](PROBLEMS_AND_RESOLUTIONS.md)。

## 实测效果与边界

| 项目 | 本轮证据 | 准确解释 |
|---|---|---|
| 真实专家回放 | PASS；完整模型/回放误差0 | 一个真实 decode 行的组件正确性，不是全模型质量资格 |
| 正常剖析 |242 对 NPU event 时长；无事件丢失、观察器错误 | 采样 layer0/3，非48层完整互斥成本表 |
| Prefill 专家准备 |5-token 请求65次准备，累计 Host60.2026s；64-token 请求274次，累计264.2664s | 准备边界合并了读取、格式处理和传输等待，不能称为纯 DMA 时间 |
| 取消/复用 | 两个32-token chunk 后取消，后续 chunk 未提交，复用成功 | 验证本次取消交错下的线性化与收口 |
| 取消响应 | 控制发送到 CANCEL_SEEN339.116509s | 仍等待在途 chunk；低延迟取消没有完成 |
| 数值参考 | CPU partial相对 L2约1.173e-4，NPU约8.817e-8，合并约3.056e-4 | 对不同存储精度分别设参考，没有把 MXFP4 当作 BF16 完全一致 |

当前状态为 **PARTIAL_ATTRIBUTION**。CPU worker 内部 packing/GEMM、DRAM 带宽和有校准的跨流重叠仍未知。首次 decode parent 的较大时长没有被内部子 span 完整解释，保留为未归因成本。所有 provider 时长为包含子调用的区间，不把父子项相加计算瓶颈百分比。

短请求 `Answer with exactly OK.` 实际返回 token271（换行）；本轮只用它验证与基线一致，不能声称已通过已知答案质量 canary。CMMLU、正式性能、真实长上下文和部署资格仍为 PENDING。

## 本轮关键修复

- 观察器事件失败不得跳过模型调用或覆盖原异常：增加回归，故障时关闭后续设备计时并降级证据状态。
- 空/坏 trace 不得假装有归因结果：补充覆盖、格式、丢失事件与错误判定。
- 一个输出 token 不一定进入 decode：保留正常 EOS，要求真实 decode 捕获，否则明确缺失。
- 捕获校验补齐请求、行来源、布局、模型/bank 身份及大小边界，并记录 placement/allocation epoch。
- 完整模型实际 Q32 驻留为0–31，旧合成测试是间隔16的32个专家。物理回放前修正 replay 分支，保留原合成 fixture；未热改运行中的 v63。
- NUMA 准入不足时，只在自有进程停止后对已记录179个项目文件执行此前批准的 DONTNEED，重新准入；没有降低48GiB门槛或全局清缓存。

## 决策与资源收口

本轮不修改调度/计算行为。下一步唯一提案是将专家 shadow 准备分解为原始 BF16 materialization、ND/NZ parent 构建、copy/retire，并核对 shadow 身份及寿命。它尚未启动，具体验收见 [实验决策](NARROW_EXPERIMENT_DECISION.md)。现有证据不足以优先改 PLE launch/wait 或重写 CPU/NPU kernel。

使用预算：2/2 次模型启动、7/8 个 HTTP 请求、3/3 次设备剖析请求、1/2 次组件启动。全模型最大自有 NPU 进程显存26550MiB，最大板级29317MiB；回放分别1060MiB和3882MiB。未触发 guard 停止。

所有本轮 Round9 容器已停止/退出；训练容器仍运行。本轮训练 NPU 用量采样为0，因此不宣称通过活跃训练争用压力测试。仍保留共享模式及15GiB训练预算。Round7定时任务未改动，未 push、未合并 main、未打 tag。

## 交付入口

- [完整结果与边界](KT_QWEN_ATTRIBUTION_REPORT.md)
- [组件及实际调用图](COMPONENT_AND_KERNEL_MAP.md)
- [问题与修复](PROBLEMS_AND_RESOLUTIONS.md)
- [机器可读状态](NEXT_ROUND_STATUS.json)
- [资源峰值与远端完整监控哈希](evidence/resource-summary.json)
- [专家回放原始结果](evidence/v20-real-expert-replay.json)
- [取消闭环原始结果](evidence/v63-cancel-result.json)
- [49项测试结果](evidence/final-related-tests.log)

远端证据保留于 `/mnt/data1/kt-round9/evidence/20260922-kt-mainline-v1`，原始 v62/v63 及组件 v20 分目录保存。v63 观测构建和修正映射后的 replay snapshot 分开留存身份清单，避免将不同源状态混为同一冻结版本。
