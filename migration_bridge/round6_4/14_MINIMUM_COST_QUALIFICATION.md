# Round6.4 最小成本资格协议

2026-09-16。依据用户明确授权：GSM8K 32＋CMMLU 256，最小成本质量检查及正式性能/最终资格结论。
本修订优先于原计划的大样本质量条款及六种性能形状矩阵；不追溯修改历史失败或旧证据。

## 质量范围和冻结

- D1 batch=1，Deferred OFF/ON 配对，各288请求，共576请求。
- GSM8K测试32；CMMLU测试256，覆盖67学科。不是已经观察过的32题训练开发集。
- 从原v2冻结测试池抽样，SHA256(6404:id)排序，不读取预测或按正确率选题。
- CMMLU每学科至少1题，余量按原冻结池学科规模最大余数分配。
- 复用v2提示词、BOS、原token IDs、EOS/stop、512/32预算、tokenizer兼容层和计分器；不额外开启few-shot调优轮。
- 输入SHA：`de64f796d530b649b7e1dc7854aebef441be65cdf60762a571305450ad7fbc7c`。
- Manifest SHA：`55b7bda2413b510203f6208444a36d1cf538987fe4cc1cc363930f60091c7566`。
- 精确输入、ID及分配见 `fixtures/fast-quality-v1/`。数据来源/许可沿用quality-draft-v2-manifest.json。
- 输入及规则已冻结；启动前仍需绑定开发集审查、实际运行源码和评测器hash。旧1319/2048 runner不可直接用于本协议。

## 质量判定

各benchmark分别要求 observed accuracy(ON)-accuracy(OFF) >= -0.02。
整数判定：100*(OFF正确数-ON正确数) <= 2*N，不浮点四舍五入。

| 集合 | 1题对应分数 | 允许净减少正确题数 |
|---|---:|---:|
| GSM8K32 | 3.125pp | 0 |
| CMMLU256 | 0.390625pp | 5 |

无效/截断计错并逐题报告；请求错误、非有限、缺失/重复ID不允许用分数掩盖。
保存correct counts、discordant pairs、invalid、truncation、配对区间。区间只作诊断，不再要求统计非劣。
不自动扩大样本；不换题；不合并两benchmark掩盖单项下降。
低OFF正确率及高截断必须披露，即便点估计通过也不能宣称广泛任务质量保持。
结论只能是 QUALITY_SCREEN_PASS/FAIL，不能称总体2pp非劣已证实。

## 正式性能的最小实现

- 一个预先指定形状：192输入tokens、256输出tokens、batch1。
- 3个ABBA块（A=OFF、B=ON），12个phase；每phase固定2次warmup和1次计时请求。
- 固定长度、相同输入；capture/observer关闭；记录TTFT、decode tok/s、总延迟、RSS/HBM及争用。
- 不跑额外六形状矩阵，不做batch4修复或新的拓扑优化。
- 用块内配对speedup汇总，报告3个block和t区间。区间下界>1且无资源混杂才称SPEEDUP_VERIFIED。
- 未达证据要求则 NO_MEASURABLE_SPEEDUP 或 RESOURCE_CONFOUNDED，不为通过追加择优轮。
- 性能样本必须验证Deferred实际执行，不能拿没有延迟命中的请求声称Deferred加速。

## 最终资格范围

复用本轮同源码/配置已完成的系统、同路径、oracle、专家、生命周期证据；若源码变化再补受影响回归。
保留必要的held-out/free-generation与稳定性审计；不再添加新的大质量campaign。
输入冻结不等于A3测试完成，性能工具仍须适配3块/单形状并校验后运行。

- Implementation PASS＋Quality screen PASS＋Speedup verified：限定为D1 batch1小样本资格通过。
- Implementation PASS＋Quality screen PASS但无明确加速：实现/小样本质量通过，未证明加速；不能标完整加速READY。
- 系统或同路径失败：BLOCKED；质量点估计失败：QUALITY_SCREEN_FAIL。
- Deferred仍默认OFF；不声称batch2/4、全模型质量或其他模型已资格化。

## 时间预算（条件估算）

按当前GSM开发集约47.6秒/题，64次请求约51分钟；CMMLU512次约5分钟。
质量连同启动/审计暂按1–1.5小时。正式性能另计模型启动时间与资源争用，不能把质量时间当成本轮全部剩余时间。
最小性能与剩余证据处理的总耗时应在适配工具后根据实际phase计时更新，不承诺未经测量的完成时刻。

## 当前验证

冻结脚本2项本地单测通过：顺序不影响选样、67学科覆盖、288唯一样本和2pp离散边界。
本协议尚无正式A3质量或性能结果，禁止预先写PASS。
