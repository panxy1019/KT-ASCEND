# Round6 当前证据状态（不得把历史失败改写为通过）

## 已完成

- Round6.4：Deferred 已实现，但未证明加速；默认关闭。
- Round6.5：four-R32 超出原冻结数值包络；历史状态 BLOCKED。
- Round6.5A：证据支持 CPU/NPU 算术扰动经层间与 router 边界放大；
  未发现对应 frozen failure 的实现不变量违例。
- Round6.6 V1：三路 backend 分解608例、128个CPU expert实例通过；
  QD6/HD6、同路径确定性、重启与系统语义审计完成。
  独立质量 screen 未通过：GSM8K3→2/32，CMMLU145→142/256。
- Round6.6A：全部8个质量得分差异案例（含2个提升）完成回放；
  32组真实 mixed partial、最终 merge 和10次重复均与生产采集 exact。
  没有找到支持修改 production 算术的新增实现缺陷。

## 当前进行中

Round6.6B：一次独立的 base-model 评测协议研究。
新测试题、固定 train/dev few-shot 示例、明确候选字母 likelihood 评分，
所有输入与规则在运行前冻结。新 All-NPU 基线为 GSM8K14/32、CMMLU174/256；
Hybrid 配对采集尚在执行，不能提前判 PASS。

这不是 V1 重评分，不是调整 B_pair，不是换 expert，也不是反复换种子寻求通过。
如果该次研究失败，将保留失败，不继续随机换题直到出现好结果。

## 不得声称

- 当前已完成全部 heavy-offload qualification。
- 原 B_pair=2.1875 在 four-R32 上通过。
- 小样本观测无净损失等同总体统计非劣。
- all-MoE、Dynamic、Deferred 加速或生产性能已经 READY。

实际本轮运行目标为110.129.0.22、Ascend910B3，不混写为历史A3硬件验证。
NPU0、32workers、four-R32、生产 overlap 保持不变。

详细证据分别位于 `round6_6/`、`round6_6a/`、`round6_6b/`；
原始大张量保留在目标服务器，Git 中有逐文件内容 manifest 的摘要与存储说明。
