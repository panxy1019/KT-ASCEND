# Round16 GitHub 发布说明

本次发布包含 Round16 的实现源码、46 个测试模块、八篇专题文档、最终状态与基准、A 线归一化关键路径证据，以及 B 线从 Physical-Q32 构造到完整模型和四进程请求对照的关键回执。

## 纳入范围

- `round16/`：共享合同、A provider/observer、B plugin/loader/runner/lifecycle；
- `round16_integration/`：真实镜像绑定、Q32 V1–V4、三页 KV、V5 smoke/performance 与比较器；
- `tests/round16/`：全部 Round16 单元与回归测试；
- `docs/round16/`：范围、A/B、KV、性能、资源、代码地图；
- `artifacts/round16*`：最终报告、状态、指标、关键阶段回执和资源退休记录；
- `tools/`：Round16 admission、A1 driver 和自有进程监控工具。

## 未纳入 Git 历史的大文件

- A1 四个可再生的 observer/profile JSONL，单文件约 58–61 MiB；
- A1 旧 Round13 span、monitor/server 原始大日志；
- B 请求块的 monitor、context-reference event、server log 和 weight trace；
- `py-spy` wheel/解包二进制；
- 容器中复制的第三方 vLLM/vLLM-Ascend donor 源码快照；
- 模型 checkpoint、MXFP4 bank、Python cache。

A1 原始 trace 的文件名、大小和 SHA-256 记录在 `GITHUB_PUBLISH_CONTENTS.json` 的 `omitted_large_evidence` 字段；原始归档身份还保留在各块的 `round16-a1-archive-sha256.json`。B evidence 的原始 `SHA256SUMS` 作为归档索引保留，发布内容是否实际纳入以本发布清单为准。

## 校验

```bash
sha256sum -c artifacts/round16/GITHUB_PUBLISH_CONTENTS.sha256
python3 -m unittest discover -s tests/round16 -p 'test_*.py'
python3 -m compileall -q round16 round16_integration tools tests/round16
```

`GITHUB_PUBLISH_CONTENTS.sha256`校验的是 JSON 清单本身；逐文件哈希位于该 JSON 的 `included_files` 中。
