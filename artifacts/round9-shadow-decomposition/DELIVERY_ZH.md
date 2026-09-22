# 08修订执行与09并行审计：本次交付

**08未完成物理验收，状态BLOCKED_WITH_EVIDENCE；09已完成本轮限定的已有证据审计。**

已修订预算、真实BF16 shadow/reference路径、原同步/临时分配、缓存影响、PLE适用范围和分维度结果状态。主执行者完成默认OFF观测及两个受控组件启动入口；gpt-5.6-sol medium子智能体完成provider审计、09覆盖账本和独立代码审阅。

63项本地相关测试、78项无NPU冻结镜像CPU测试通过。独立审阅的5项重要问题均已修复，新增回归先失败后通过：OFF/ON身份一致、分阶段结果、异常降级、A/B关联、projection+workspace峰值。另补充实际OFF/ON输出比对、tensor storage元数据及冻结provider检查。

**物理阻塞：** 所选160个CPU在NUMA2/3/6/7，量化idle约40.6%，未达到既有>80%门槛；正式OFF启动前仍拒绝。卡15没有NPU进程不代表CPU准入通过。未放宽门槛、未改CPU集合、未清缓存、未停止其他任务。本轮物理启动0/2，模型/HTTP均0；A/B数值、phase成本和设备生命周期均NOT_RUN，不继承上轮回放PASS。

09覆盖A01–A10、B01–B09，分别标注当前证据、历史未重验、未运行。v61基线、v63观测、v20replay三种source身份分开。质量canary仍PROPOSED_NOT_RUN；旧max_new_tokens1返回换行只能证明路径一致。没有进入M2行为优化或M3资格，也没有追加设备预算。

资源收口已确认无本轮运行容器，CPU-only检查正常退出，训练及现有外部容器未修改；无新后台监控、定时任务修改、push/merge/tag。原始暂停handoff文件未修改。

## 接续与证据

- [实际接口与身份合同](SHADOW_PREPARE_CONTRACT.md)
- [实现、测量口径与物理阻塞](SHADOW_PREPARE_DECOMPOSITION.md)
- [问题与修复](PROBLEMS_AND_FIXES.md)
- [机器可读执行结果](shadow-decomposition-result.json)
- [下一步唯一决策](NEXT_SINGLE_CHANGE_DECISION.md)
- [09覆盖矩阵](AB_COVERAGE_MATRIX.md)
- [质量canary合同](CANARY_CONTRACT.md)
- [冻结源清单](SOURCE_SNAPSHOT.json)

恢复到既有资源准入后，从E3冻结OFF→ON组件执行接续；仍需重新验证CPU/NUMA/HBM及训练完整容器ID，不重复开发或重新运行已完成离线检查。所有设备比较使用同一source/capture/sidecar身份。若source需变更，建立新snapshot并重审比较身份，不热改当前冻结目录。
