# 本轮问题、修复与未解决项

1. **资源表混淆组件与全模型额度。** 修订分开共享系统上限与4%/4096MiB/64GiB组件限额；原300MiB shadow+200MiB staging+512MiB workspace均计账，零新增指不增加原有对象外的缓冲区。
2. **观测与原计算的关系。** 添加最小phase边界和只读tensor metadata，不复制prepare算法。保存a5b26444e两份golden source，以AST去观测等价检查证明原计算/同步语句未改变。实际NPU等价仍NOT_RUN。
3. **本地缺torch。** 扩大适配器回归时一个测试模块collection失败，原因是本地Python没有torch。保留日志，使用唯一一次无NPU设备的冻结镜像CPU检查；78项PASS。没有为此启动NPU或安装/升级依赖。
4. **OFF/ON身份漂移漏洞。** 独立审阅发现只检查OFF状态不足。新增RED→GREEN测试，ON启动前要求source manifest、capture tensor、sidecar三者完全相同；capture/sidecar额外固定为上轮已知SHA。probe校验实际unquant provider SHA和torch版本。
5. **结果缺少分阶段数据。** 原汇总只有parent覆盖/残余，不能回答准备阶段成本。RED→GREEN增加按phase的inclusive host及union时长、独立device列表；显式说明嵌套phase不能相加。
6. **异常span误标有效。** 原summarizer未检查outcome。RED→GREEN后异常parent/child降级INCOMPLETE；观察器错误不会替换原provider异常或跳过provider。
7. **A/B事件关联不足。** 增加每事件run/case/request/placement/allocation/shadow-lifetime关联，RED→GREEN验证子事件继承。捕获输入驱动的BF16 shadow与旧mixed输出分开，不混称prefill capture。
8. **执行峰值漏算ND staging。** 审阅发现baseline后峰值同时包含projection和workspace。改为max_projection_bytes+observed_workspace_bytes；用历史616,490,496bytes构造RED→GREEN回归，避免合法执行被512MiB误拒。没有借此扩大allocator或owned限额。
9. **物理OFF/ON需要实际对比。** 独立reference通过不自动证明观察一致。追加RED→GREEN测试，ON必须与OFF对应case输出hash、prepare计数和logical copied bytes一致。
10. **所选CPU忙碌，未解决。** 量化idle约40.6%，未达到>80%；正式launcher在创建容器前再次拒绝。保留BLOCKED_WITH_EVIDENCE，无NPU启动、无清缓存、无干预其他任务。本轮不能交付分阶段实测或数值PASS。

独立审阅由gpt-5.6-sol medium执行。所有5项重要发现均已修复并通过回归；没有另发重复审阅。硬件数值、设备事件完成、分配回归和OFF/ON实测被审阅者明确保留为未知，本轮状态照此记录。09文档的历史继承标签与三份源码身份也已分开审计。
