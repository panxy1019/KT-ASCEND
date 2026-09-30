# Round18 交付导航

Round18 研究的问题是：在保留 vLLM-Ascend 原生 `AscendMoERunner`、router、
shared expert、finalize 和 Graph 执行组织的前提下，能否只把 routed-expert
容量管理交给 KT，并在同资源 TP1 下形成可晋级方案。

最终状态为 `ROUND18_COMPLETE_NO_PROMOTION`。Native Q32 partial 与最小
CPU complement 在 eager 下正确，但冻结栈没有形成 graph-safe 的请求候选；
Native expert pool 又被 current-token device planner、Q32 NZ29 固定槽写入和
原子 mapping publication 合同共同阻断。默认配置继续使用 Round17 S0。

## 首要入口

| 文件 | 用途 |
|---|---|
| `ROUND18_FINAL_REPORT_ZH.md` | 完整中文结论、关键数据、边界和八个问题的回答 |
| `FINAL_STATUS.json` | 机器可读总终态 |
| `FINAL_CONFIG.json` | Round18 后保留的默认配置 |
| `CURRENT_EXECUTION.md` | 终态、重新开启条件和资源状态 |
| `ROUND18_BENCHMARK.csv` | 请求性能与 `NOT_RUN` 边界的统一表 |
| `EVIDENCE_SHA256.json` | 源码、测试、文档和原始证据的 detached SHA-256 清单 |

## Phase I：Native-preserving Hybrid

| 文件 | 用途 |
|---|---|
| `R18_RESOLVED_MOE_SEAM.md` | 冻结 Native MoE seam、所有权和调用边界 |
| `R18_NATIVE_PARTIAL_MOE.json` | 512 logical → Q32 physical resident partial 真机结果 |
| `R18_HYBRID_EAGER_COMPONENT.json` | Native partial 与 C128/MXFP4 complement 组件结果 |
| `R18_HYBRID_EAGER_RESULT.json` | 完整 48 层 eager smoke、取消和 reuse |
| `R18_PHASE_I_DISPOSITION.md/.json` | PIECEWISE、Breakable、FULL-compatible 的最终去留 |
| `R18_GRAPH_MODE_MATRIX.json` | 各 Graph mode 的真实 resolved 状态 |
| `R18_GRAPH_CORRECTNESS.json` | Graph 正确性硬门 |
| `R18_GRAPH_REQUEST_COMPARISON.json` | 没有 graph-safe 候选时请求 ABBA 保持 `NOT_RUN` |

## Phase II：Native expert pool

| 文件 | 用途 |
|---|---|
| `R18_ROUTE_WORKING_SET.json` | fresh W1/W2/W4 route、Q32/Q64/Q96/Q128 工作集分析 |
| `R18_PHASE_II_POOL_ECONOMICS.md/.json` | 容量、miss 流量和可部署 policy 裁决 |
| `R18_BACKING_FORMAT_MATRIX.json` | BF16/NZ29、independent list、预处理 blob、W8A8 去留 |
| `R18_EXPERT_POOL_COMPONENT.json` | v017/v018 fixed-slot 组件结果 |
| `R18_NZ29_FIXED_SLOT_SOURCE_AUDIT.md/.json` | 当前 Q32 parent writer 边界 |
| `R18_EP1_INDEPENDENT_EXPERT_STORAGE_SOURCE_AUDIT.md/.json` | independent list consumer seam |
| `R18_EP1_INDEPENDENT_LIST_PROBE_CONTRACT.md/.json` | 保留的离线 bounded future probe；设备未运行 |
| `R18_P10_GRAPH_SAFE_POOL_SOURCE_GATE.md/.json` | FULL replay 外 mutation seam 与缺失 planner |
| `R18_EXPERT_POOL_GRAPH_COMPAT.json` | Phase II graph compatibility 终态 |
| `R18_EXPERT_POOL_REQUEST_COMPARISON.json` | 无可部署候选时正式请求对照保持 `NOT_RUN` |

## 资源与原始证据

- `ROUND18_RESOURCE_RETIREMENT.json`：最后设备块、缓存退休和远端清场回执。
- `evidence/`：每次组件、完整模型、Graph 诊断、route trace 和 fixed-slot
  探针的原始日志、准入、容器与结果文件。
- `evidence/final-remote-clearance/`：最终远端进程、容器及物理12只读快照。

`evidence/` 中的 FAIL 记录是研究过程的一部分，不应删除或覆盖。早期阶段审计
若与最终终态不同，会明确标记为历史快照或由后续终态文件 supersede。

## 代码与测试

- `round18/`：纯合同、Native seam、Graph 边界、route/pool 分析和发布协议。
- `round18_integration/`：冻结镜像中的受控设备/模型入口。
- `tests/round18/`：71 项离线回归测试。

Round18 没有修改默认生产配置。任何后续工作若要重新开启 expert pool，必须先
提供 current-token device-side miss planner，以及 format-aware fixed-slot writer
或经过设备验证的 parentless independent-slot owner；不能返回 PIECEWISE 逐算子
补丁、完整 NZ parent 重建或每 token Host route readback。
