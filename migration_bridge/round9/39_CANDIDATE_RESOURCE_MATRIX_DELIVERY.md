# Round9 C128 全模型资源矩阵交付

更新：2026-09-21 01:26 UTC。目标：110.123.0.3，独占物理卡15。

**六个组合均完成全模型资源/系统门禁；五个组合同时通过严格内容检查，BF16 Dynamic 保留严格 JSON 失败。** 本轮一次性执行已结束并释放设备。

| CPU 专家存储 | 放置策略 | 运行 | 资源/系统 | 严格答案 | 进程 HBM 峰值 MiB | 宿主容器峰值 GiB |
|---|---|---|---|---|---:|---:|
| BF16 | static | [v10](evidence/physical15/20260920-full-model-v10/trial-summary.json) | PASS | PASS 4/4 | 39856 | 288.11 |
| MXFP4 | static | [v19](evidence/physical15/20260920-full-model-v19/trial-summary.json) | PASS | PASS 4/4 | 39854 | 107.28 |
| MXFP4 | request_boundary | [v20](evidence/physical15/20260920-full-model-v20/trial-summary.json) | PASS | PASS 4/4 | 41060 | 131.03 |
| Q8_0 | static | [v21](evidence/physical15/20260920-full-model-v21/trial-summary.json) | PASS | PASS 4/4 | 39854 | 250.45 |
| Q8_0 | request_boundary | [v22](evidence/physical15/20260920-full-model-v22/trial-summary.json) | PASS | PASS 4/4 | 41060 | 250.56 |
| BF16 | request_boundary | [v23](evidence/physical15/20260920-full-model-v23/trial-summary.json) | PASS | FAIL（JSON 围栏） | 41062 | 236.51 |

每组独立完成模型/银行身份绑定、CPU/NUMA/内存准入、真实首次生成、8次系统请求、实际3840输入+256输出、主动中止及状态复用。实际128个工作线程分别绑定128个不同CPU。静态检查含精确重复；Dynamic 检查含有限输出与请求边界清理，不能据此宣称完整模型数值等价。

Dynamic 三组各完成18代事务，覆盖全部16个测试请求和2个健康请求；专家复制次数：MXFP4 22072、Q8_0 21740、BF16 21440。所有48层的计划、映射连续性和复制计数均有原始事务记录。

BF16 Dynamic 第四项 canary 输出了包含 name=apple、count=3 的对象，但额外包裹 Markdown JSON 代码围栏，因此严格解析失败。已捕获完整原始输出，未修改提示词或放宽解析器。前三项正确且全部数值有限。其整体状态为 FAIL_CANDIDATE_CONTENT_CANARY；矩阵为 COMPLETED_WITH_CANDIDATE_FAILURES，并以非零状态退出。尚不能把格式差异的根因归结于模型或迁移数值实现，仍需相同放置/历史的完整模型数值对照。

本轮关键修复已进入独立候选路径：ARM Q8 激活布局、低比特160/320切片尾部、低比特F32分区归并、NZ父张量批量写入、独占模式分配器上限与监控绑定、严格答案原始响应保留。已通过的原生/算子门禁和历史失败见[执行详录](38_CANDIDATE_FULL_MODEL_TRIALS.md)。

资源配置为四个NUMA子池各32线程，辅助CPU预算8，总quota136；容器RAM上限640GiB。独占模式分配器42GiB、进程保护44GiB、准入模型预算45GiB，并额外保留8GiB及实测系统基线。这些结论仅适用于本次独占准入；重新共享时必须重新核算，不能直接沿用。

停止核验：物理15无运行进程，HBM2878MiB；Round9模型/监控/矩阵进程均退出，mzy-train及kt-r7-a3-return-v1未改动。Round7/Round9自动任务保持PAUSED。没有性能测试、push、main合并或tag。

未取得的资格：完整256题精度及基线对比、候选重启等价、Dynamic完整模型相同放置数值oracle、性能和部署资格。BF16 Dynamic严格格式问题仍未解决。本交付仅表示开发框架和本轮完整模型资源验证已落地，不表示Round9全部精度资格完成。

证据入口：[绑定索引](evidence/physical15/20260920-candidate-matrix-v4/delivery-index.json)、[矩阵状态](evidence/physical15/20260920-candidate-matrix-v4/status.json)、[最终设备快照](evidence/physical15/20260920-candidate-matrix-v4/final-device-state.json)。索引绑定六份summary、对应audit、审计程序和矩阵脚本的SHA，原始文件保持不改写。
