# Round6.4 系统回归与 oracle

## 新执行的 A3 preflight

Evidence：`A3_VERIFIED`，仅限下列基础测试。

| 测试 | 结果 |
|---|---|
| Round5 framework transfer/callback（含 BF16/F32/int64 各 1000 往返） | 5 PASS |
| Deferred selection | 7 PASS |
| SGLang Ascend alignment contracts | 7 PASS |
| 合计 | 19 PASS |

实际耗时：pytest 11.12 秒；guard 返回 EXIT_ZERO，未发现匹配的运行异常。
原始输出：`evidence/r64-preflight-v1/guard/process.log`、`junit.xml`。
来源：`evidence/r64-preflight-v1/provenance.json`。

guard result SHA256：`08b163491dbc37e86149af2ebf45e6441c267cd73ecaff7c455d5c05a2ca9b05`。
provenance SHA256：`f5bcbda2a3b9e751c2d6a622af625ddbf0ab281d2ba88f3291339f9e7ea39f84`。

该结果不代替完整 R2A expert 安全门禁、D1 S6/restart/oracle/lifecycle 或质量测试。

## Preflight 启动诊断说明

v1 在 provenance 阶段进行大 GGUF 哈希，尚未出现输出文件时被误认为可能未启动。
因此曾开启 v2 启动诊断。确认 v1 健康后，精确停止本项目 v2 shell 及其独立
metadata 子进程；v2 未进入 pytest/模型请求，不作为任何证据。空目录保留。
不再凭“结果文件尚未生成”启动重复任务，应先检查所属 PID 与当前命令。
未操作其他任务，v1 自然完成。

## Compact 控制

工具：`tools/run_compact_controls.py`；本地三个针对性测试 PASS。
A3 root：`/workspace/results/r64-compact-v1`。
顺序：P1_B0 → P2_B0 → P3_B0 → D1_NODEFER → D1。
固定 S6 四个 ID，每 profile 20 请求：32、32、8、16、32 tokens。
observer OFF，route recorder ON/cache OFF，与旧 compact 控制配置一致。
P1/P2/P3 同时对比已有相同 profile 的 32-token reference；D1 仅检查自身重复。
这不是性能测试，也不是全部 D1 runtime。

状态：RUNNING；以各 profile `report.json` 和最终 `progress.json` 为准，未提前填 PASS。

## Oracle 复用边界

Round6.3 两个失败历史的 production/private oracle 结果仍可作为既有机制证据，
不改为 Round6.4 全面通过；另四个预先指定工程请求尚需运行。
S6 156 请求、独立 restart、全请求 no-hit、1000 真实 expert forwards 尚未关闭。
