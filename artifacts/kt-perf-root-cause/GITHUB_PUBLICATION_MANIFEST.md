# GitHub 发布范围

本文件记录 `codex/kt-root-cause-closure` 分支的紧凑发布边界。完整原始证据继续保存在本地工作树和远端冻结目录，Git 提交保存结论、代码、测试、汇总结果、可重建账本所需的紧凑输入，以及所有原始文件的 SHA-256 清单。

## 发布内容

- 本目录顶层的最终报告、状态、原因卡、token/gap 账本、资源/测试回执和 `EVIDENCE_SHA256.json`；
- `tools/kt_perf_closure/` 与 `tests/kt_perf_closure/`；
- v506、v507、v509 的 `kt-root-cause-analysis-v2.json`、profile result、region summary、source/运行身份和冻结 evidence manifest；
- v509 的 `round13-spans.jsonl`，用于与 `kt-root-cause-analysis-v2.json` 重建 `DEPENDENCY_GAP_ACCOUNTING`；
- admission v13、native import probe 和小型源码样本。

## 未放入 Git 历史的原始大文件

- 三个块各约 58 MiB 的 `round16-a1-profile.jsonl`；
- 三个块各约 8.4 MiB 的原始 `native-task-trace.jsonl`；
- v506/v507 的原始 `round13-spans.jsonl` 和旧版 analysis；
- 高频 monitor、容器日志和与最终结论无关的传输归档。

这些文件没有被删除或改写。它们的路径、大小与 SHA-256 均在 `EVIDENCE_SHA256.json`，对应远端冻结目录为：

- `/mnt/data1/kt-round9/evidence/20260929-kt-root-cause-profile-a-v506`
- `/mnt/data1/kt-round9/evidence/20260929-kt-root-cause-profile-b-v507`
- `/mnt/data1/kt-round9/evidence/20260929-kt-root-cause-full-host-a-v509`

## 保密与容量边界

提交不包含模型 checkpoint、MXFP4 bank、容器镜像、密码、API key 或全量环境变量。v506 的原始 evidence manifest 末尾异常字面量 `\\n` 原样保留；规范化副本与原始 SHA 见 `RAW_EVIDENCE_PRESERVATION.json`。
