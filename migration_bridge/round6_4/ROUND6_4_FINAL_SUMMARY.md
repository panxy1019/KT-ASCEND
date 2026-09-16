# Round6.4 最终总结

日期：2026-09-16。**本轮已完成，结论为 IMPLEMENTED_NO_SPEEDUP。**

本轮按用户批准接受官方跨层Deferred作为近似计算，并采用GSM8K32＋CMMLU256、
每项观测准确率下降不超过2pp的最小成本筛查。未恢复旧的跨模式token-exact门禁，
也没有删除难题、修改专家、放宽同路径确定性或expert安全阈值。

## 最终门禁及适用范围

| 项目 | 结果 | 证据 |
|---|---|---|
| 原平台系统、oracle、exact-once、生命周期 | PASS |03–08、11文档|
| 原平台D1同路径重复及重启 | PASS |06文档|
| 缩减质量筛查 | PASS，非总体非劣证明 |15文档|
| 原平台正式性能 | NO_MEASURABLE_SPEEDUP |16–17文档|
| 原平台2304token稳定性 | PASS，观察到平台化 |18文档|
| 新平台基础传输 |19/19 PASS|21文档及migration证据|
| 新平台CPU回归 |18PASS，3历史排除|21文档|
| 新平台真实专家 |16/16PASS，最大rel-L2 0.003913|21文档|
| 新平台16/32线程compact |各32/32PASS；跨线程32/32轨迹相同|21文档|
| 新平台三模式完整性 |408请求、26520forward，全部PASS|19文档|
| 新平台最终离线分析 |COMPLETE，全部指标有限|19文档|
| 新平台完整质量/正式性能 |NOT_QUALIFIED，未重跑 |不得引用原平台结果替代|
| 新平台完整内存曲线 |不具备；用户暂停了采样 |仅保留部分样本|
| 加速发布 |**NOT_READY** |20_FINAL_ADR.md|

## 质量与性能结果

原平台110.123.0.3、16CPU线程、D1/batch1：

| Benchmark | OFF | ON | Delta |
|---|---:|---:|---:|
| GSM8K32 |5/32|5/32|0pp|
| CMMLU256 |139/256|139/256|0pp|

满足批准的观测2pp筛查，但GSM基线仅15.625%，且OFF21/ON18题截断；
CMMLU两侧均7题invalid，全部按错误计分。不得宣称普遍质量无损或总体非劣。

正式性能为冻结192输入/256输出、batch1、3个ABBA block、12个独立进程：
配对加速比**0.989243×**，95%CI **[0.912743,1.072153]**。
未达到下界大于1的门禁。不代表所有负载都不可能加速；本轮不追加试验寻找好结果。

## 新服务器迁移与CPU扩容

110.129.0.22，Kunpeng920＋Ascend910B3，仅NPU0，专用容器
`kt-r64-910b-migration`，CPU0–31。其他容器、宿主系统库及其他任务未修改。

迁移中解决两项环境问题：容器补齐官方CANN9.0的910B算子包；原扩展启用
目标CPU不支持的SVE/BF16/i8mm，遂用相同源码按armv8.2-a+fp16+dotprod重编译。
保留原始失败、新旧二进制哈希、编译参数、源码差异和回归证据。
CPU16组输出哈希与原平台逐项一致，但CPU/NPU误差指标并非全部逐字相同。

原扩展SHA：3fbb95c00c5906b905e8fd689346f4e78f86714fddb6372fa0e481e0da7ce43f。
新扩展SHA：0f3aac9009931b18853c9b8fd6f6bba8b02d6ea396e79b2029bf9ce092049895。

## 最后补齐的三模式采集

OFF自由生成→ON同历史→ON自由生成，各136请求、8840forward。
合计26112目标位置＋408lookahead，未知/不完整均0；15项共同源码哈希一致。
HD3自由生成分叉45/128，FD3分叉2/8，属于近似计算诊断，不用于拟合新阈值。
同历史最大logit差HD3为5.5625，FD3为3.625，指标均有限。
FD3两模式都可出现重复、截断、格式不遵循；不能由轨迹一致推导任务质量好。

旧平台中断采集保持USER_PAUSED，未与新平台拼接。原始PT保留在新服务器
`/workspace/results/r64-910b-descriptive32-v1`，元数据压缩归档及完整文件哈希
在`evidence/final-descriptive-910b/`。详见19文档。

## 代码、资格与后续边界

本轮未修改Deferred生产算法；主要新增测试、审计、平台适配和文档。
默认Deferred仍OFF。Batch2只有工程验证；Batch4历史同路径失败仍不合格。
TP2、Graph、Dynamic Placement、其他模型、量化和生产并发均不在本轮资格范围。

实际测试parent：e3084d309d6f39b8a9f7185d36ac27e4b77526d7，
child：eaa58ccea4ee372b8b207dd16f398eafe3da37ff，另有显式记录的源码overlay。
本地文档分支HEAD不能冒充A3实际加载源码；逐文件哈希记录在各run provenance。
本地分支：`codex/round6-4-official-deferred-approximation`。

最终状态：

```text
ROUND6_4_EXECUTION = COMPLETE_WITH_EXPLICIT_SCOPE_LIMITATIONS
SOURCE_D1_QUALITY = QUALITY_SCREEN_PASS
SOURCE_D1_SPEEDUP = NO_MEASURABLE_SPEEDUP
ROUND6_4_CLASSIFICATION = IMPLEMENTED_NO_SPEEDUP
TARGET_910B_MIGRATION_AND_DESCRIPTIVE = COMPLETE
TARGET_910B_FULL_QUALITY_AND_PERFORMANCE = NOT_QUALIFIED
DEFERRED_APPROX_ACCELERATION_READY = NO
DEFAULT_DEFERRED = OFF
```

不创建加速READY标签。不自动开启下一轮性能改造；先以本轮结果作为可审计基线。
