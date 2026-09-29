# KT 最终性能根因收口：执行终态

日期：2026-09-29
状态：`KT_ROOT_CAUSE_CONFIRMED`
默认配置：冻结 S0，不晋级新候选
设备任务：已退出；物理 12 无本项目进程
旧定时任务：未恢复

## 已完成

1. 恢复 Round16 S0 的模型、CPU bank、Q32 map、native、provider 和请求身份。
2. 用两个 fresh 进程 v506/v507 直接连接 native enqueue、worker start/complete、TaskQueue sync、48 层 Expert parent 与 scheduler batch。
3. 用 v509 将 Host 父调用覆盖扩展到全部 36 个 GDN、12 个 QSA、48 个 Routed Expert 和 PLE consumer，形成 125 个稳态 Decode token 的顶层墙钟分区。
4. 将当前直接时序与 Round11 真实 CPU fixture、Round15 QSA selector 和 resident-MoE graph 的完整请求干预连接起来。
5. 验证当前 CPU Expert 约 5.5 ms/token 的服务已经被同层 NPU 工作隐藏，真正暴露等待约 0.13 ms/token；排除 C128 算术是当前约 3.x tok/s 的主因。
6. 用相邻父调用边界把 98.47 ms/token residual 进一步分为 attention→Expert 51.14 ms、Expert→下一层 attention 32.06 ms、scheduler/PLE 边界 15.27 ms；不把异步区间误写成纯 Host 时间。
7. 冻结原始证据、资源回执、测试结果、原因卡、中文报告和最终状态。

## 最终判断

慢因来自 48 层串行执行中反复发生的 NPU 主干、Routed Expert 内的 resident-NPU/dispatch/glue、完整 QSA、GDN，以及 Host 小段下发和依赖边界。它不是一个单独的 CPU kernel、callback 或可直接删除的同步。

本轮没有得到可晋级修复。继续优化 Kunpeng microkernel、TaskQueue timeout、QSA selector 单点或 resident-only graph 的预期请求收益都很小。若开启新项目，边界应是完整 QSA parent 或覆盖更多稳定 NPU 区域的 exact decoder segment，并重新完成 state、取消、资源和请求 ABBA。

## 证据边界

- v509 的全层观察会增加墙钟，因此部署速度固定使用 profiler-off v403/v404；
- 顶层墙钟和 residual 的依赖位置均已完整分类，但 33.88% 区间内的 Host active、queued device execution 和 stream wait 没有继续拆成单 kernel；
- 没有 PMU 数据，也没有在本轮生成新的 detached hidden/route fixture；
- Round15 两项请求干预是方向性配对，不是四进程 ABBA。

这些限制不会改变本轮最关键的直接时序事实：18,000 个层调用中，CPU 均在消费者 TaskQueue wait 前完成。
