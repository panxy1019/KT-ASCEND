# KT W2 性能根因证据包

本目录是冻结 Round16 S0 下的最终性能根因收口。建议按以下顺序阅读：

1. `FINAL_EXPLANATION_ZH.md`：完整中文结论；
2. `ROOT_CAUSE_CARDS.md`：按原因与去留整理的决策卡；
3. `FINAL_STATUS.json`：机器可读终态；
4. `TOKEN_ACCOUNTING.json/csv`：v509 的 125 个稳态 token 顶层分区；
5. `DEPENDENCY_GAP_ACCOUNTING.json/csv`：其余 98.47 ms/token 的相邻依赖位置分解；
6. `TASK_WAIT_CAUSAL_LINKS.json`：CPU submit/task/sync/merge 的直接因果连接；
7. `CAUSAL_EXPERIMENTS.json`：QSA、resident graph 和当前 CPU 边界的干预结果；
8. `RESOLVED_CODE_MAP.md`：实际运行路径、符号和 source/build identity；
9. `RESOURCE_TIMELINE_SUMMARY.json`、`TESTS.json`、`EVIDENCE_SHA256.json`：资源、验证与哈希。

原始 profile 位于 `profile-a-v506/`、`profile-b-v507/` 和 `full-host-a-v509/`。v506 的远端 SHA 清单末尾含字面量 `\\n`；原始文件未修改，规范化副本与处理说明见 `RAW_EVIDENCE_PRESERVATION.json`。
