# Round17 失败与修复账本

原始 FAIL 全部保留，未覆盖为 PASS；每次替代均有新的 run identity 和实际改动。

| Run | 失败位置 | 根因 | 请求完成 | 修复后的身份 |
|---|---|---|---:|---|
| v510 | 首个诊断请求 | 12 个 QSA layer 共享 rotary provider，被 observer 按层重复包装 | 0 | observer 改为共享 provider 只包装一次，并动态解析当前 layer；v512 PASS |
| v513 | 诊断 B 启动 | snapshot builder 保留 donor A block ID | 0 | builder 同时冻结 launcher block identity；v514 PASS |
| v515 | S0 启动后 source gate | 派生器误改冻结 budget stage 字段 | 0 | 只重置消耗计数，保留 source gate；后续身份 |
| v517 | S0 driver 前 | 容器路径漏掉 snapshot 目录 | 0 | runner 由 snapshot basename 构造执行路径 |
| v518 | S0 driver import | PYTHONPATH 漏掉 snapshot 目录 | 0 | executable path 与 module root 统一；v519 PASS |
| v519 | 请求全部完成后的打印 | postcheck 读取 `summary` 而结果字段是 `summaries` | 10/10 PASS | 只修后置校验；v519 请求证据有效，v520 runner 完整 PASS |

v515 的 `Created` 容器最初未进入 runner 的 `container_id` 变量。最终回收时根据 `container-created.json` 的完整 ID 与 `docker inspect` 精确匹配，确认未运行后删除。没有停止或修改外部容器。
