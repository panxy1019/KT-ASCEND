# ADR：接受官方跨层 Deferred 近似的资格目标

日期：2026-09-15。
Decision：APPROVED_EXECUTION_SCOPE；不是 QUALITY_APPROVED 或 RELEASE_READY。

后续范围修订：用户批准的最小成本协议见14_MINIMUM_COST_QUALIFICATION.md，
它替代下文历史的大样本统计门禁。实际小样本质量结果见15，性能结果见17。
本ADR保留审批时原文，不应据其历史规模启动旧1319/2048矩阵。

## 决定及依据

用户审批同目录完整 Round6.4 方案，并批准每个主 benchmark 2pp 绝对质量损失容忍。
本轮接受 L 层 Deferred 贡献在 L+1 合并所产生的计算图近似。
不再以 ON/OFF 全 token exact、任意候选 pair ordering 或固定 logit reserve
作为进入独立质量评估的前置条件。数值画像仍保留，角色是诊断。

这是明确改变验收目标，不是修订 Round6.3 已观察数据的标签。
Round6.3 根据原冻结协议 REJECTED，原报告及其原始证据不变。

## 不改变的系统要求

同一路径/固定请求日程的确定性、CPU/NPU ownership、route weight 与 shared expert
恰好一次、terminal/pending drain、framework copy_ 与 buffer lifetime、finite 仍须成立。
近似不能解释漏算、重复消费、跨请求污染、未同步读取、死锁或 NaN。
相同 operands 的 expert 安全门禁仍保留；不同 history 的 hidden 不作同 operands 比较。

## 质量与收益分离

同 D1、同精度、同 batch 下 ON 对 OFF 做主要 A/B。
GSM8K test 与 CMMLU stratified2048 各自验证 −2pp 非劣效界。
主质量统计实现、样本及解析协议尚待完整冻结，当前未生成主质量 B 结果。
不能把零 discordance 的退化 bootstrap 区间当作总体零风险。

实现、质量、速度三项独立记录：实现 PASS 不自动等于质量 PASS，质量 PASS
不自动等于有加速。未测到收益可结束为 IMPLEMENTED_NO_SPEEDUP；
质量证据不足结束为 INCONCLUSIVE，不自动放宽 margin 或更换题目。

## 生产变更策略

先审计已有官方机制与最小 Ascend 适配；本轮开始未改生产文件。
仅当新系统回归指出具体缺陷时做独立最小 fix，不整包搬迁动态 main。
默认 Deferred OFF；任何 READY 声明仅覆盖实际通过的 profile。
