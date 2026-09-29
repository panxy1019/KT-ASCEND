# 代码地图与测试说明

## 1. 共享合同

| 文件 | 职责 |
|---|---|
| `round16/common/contracts.py` | immutable identity、tensor spec、route batch、result lease |
| `round16/common/resource_target.py` | NPU/NUMA 资源目标与身份 |
| `round16/common/parameterized_admission.py` | 参数化 2×10 秒准入 |
| `round16/common/perf_contract.py` | workload、窗口吞吐和配对比较合同 |
| `round16/common/round15_barrier.py` | Round15 终态与父配置接续 |

## 2. A 线

| 文件 | 职责 |
|---|---|
| `round16/a/provider_manifest.py` | provider 来源、能力和 fallback 范围 |
| `round16/a/registration.py` | 幂等、无副作用注册 |
| `round16/a/sglang_adapter.py` | SGLang metadata/lease 薄适配 |
| `round16/a/source_gate.py` | source/build identity gate |
| `round16/a/fresh_a1_runtime_observer.py` | Host 边界和 deferred NPU event observer |
| `round16/a/fresh_a1_profile.py` | 两块六请求 profile 合同 |
| `round16/a/fresh_a1_evidence.py` | raw trace 归一化与 SHA 绑定 |
| `round16/a/fresh_a1_analysis.py` | 非叠加关键路径分析和 A2/A3 gate |
| `round16/a/fresh_a1_device.py` | 设备请求 driver 合同 |
| `round16/a/fresh_a1_remote.py` | 远端生命周期与证据归档 |
| `round16/a/fresh_a1_snapshot.py` | 隔离 snapshot 构建和 source identity |

## 3. B 核心合同

| 文件 | 职责 |
|---|---|
| `round16/b/model_plugin.py` | 只对目标 Qwen 模型生效的 plugin 注册 |
| `round16/b/selective_loader.py` | 逻辑512／物理Q32 load plan |
| `round16/b/selective_preallocation_installer.py` | 构造前 Q32 owner 注入 |
| `round16/b/resident_packed_loader.py` | packed checkpoint resident row 选择 |
| `round16/b/source_bound_q32_adapter.py` | 真实 routed owner source-bound adapter |
| `round16/b/source_bound_hybrid_runner.py` | selector-once、CPU/NPU split 和 exactly-once merge |
| `round16/b/allgather_q32_device_adapter.py` | Ascend ALLGATHER 物理 slot device tensor adapter |
| `round16/b/kt_expert_backend.py` | KT submit/sync/drain ABI |
| `round16/b/lifecycle_bridge.py` | request epoch、completion、cancel 和 late-result rejection |
| `round16/b/graph_boundary.py` | 图外 CPU 任务和 replay identity 合同 |
| `round16/b/exact_donor_worker_hook.py` | exact image worker lifecycle hook |
| `round16/b/native_source_preflight.py` | vLLM/vLLM-Ascend 精确源码/API preflight |

## 4. 真实镜像集成与 probe

| 文件 | 职责 |
|---|---|
| `round16_integration/b2_live_bindings.py` | exact-image owner、runner、Host PLE、QSA cache/store binding |
| `round16_integration/b2_live_worker.py` | 目标 worker 入口 |
| `round16_integration/b_q32_preallocation_live_factory.py` | V1 真实 Q32 constructor |
| `round16_integration/b_q32_loader_live_factory.py` | V2 selective loader/post-load |
| `round16_integration/b_q32_hybrid_live_probe.py` | V3 real Hybrid MoE 数值和生命周期 |
| `round16_integration/b_q32_all_layer_residency_probe.py` | V4 48层同时驻留 |
| `round16_integration/b_qsa_multihead_device_probe.py` | 三页、多 KV head、GQA sparse attention probe |
| `round16_integration/b_v5_llm_probe.py` | V5 full-model construction probe |
| `round16_integration/b_v5_smoke_probe.py` | canary、Decode、cancel、reuse |
| `round16_integration/b_v5_perf_probe.py` | B request performance block |
| `round16_integration/s0_perf_http_driver.py` | S0 对照 block |
| `round16_integration/compare_request_blocks.py` | 四进程严格比较 |

## 5. 运维工具

| 文件 | 职责 |
|---|---|
| `tools/round16_admission.py` | 资源准入 CLI |
| `tools/monitor_owned_server.py` | 仅跟踪自有容器/进程/HBM |
| `tools/round16_a1_container_driver.py` | A1 容器请求控制 |
| `tools/round16_a1_remote_block.py` | A1 远端生命周期 |
| `tools/install_round16_a.py` | A provider 安装入口 |

## 6. 测试范围

`tests/round16` 包含 46 个测试模块，最终运行 327 项测试，重点覆盖：

- contract immutability 和 lease；
- A 注册、metadata/fallback、source gate；
- A1 profile、observer、evidence normalization 和 candidate gate；
- logical/global/physical expert map；
- packed selective loader 和 alias；
- all-CPU/all-NPU/mixed/padding route；
- exact worker/plugin/source preflight；
- cancel、late completion、slot reuse；
- graph boundary 不复用 capture 时 CPU 结果；
- Physical-Q32 V1–V4 probe receipt 合同；
- QSA cache shape、HND/NHD、多 KV head store；
- 单页 shape patch 不得冒充 KV 资格；
- smoke/performance schedule 和完整 output budget；
- 四进程比较和保护门；
- S0 guard 只接受本轮自有命名。

测试输出：[FINAL_TEST_OUTPUT.txt](../../artifacts/round16/FINAL_TEST_OUTPUT.txt)。

## 7. 默认开关

所有 Round16 新路径默认 OFF。最终没有修改全局默认运行时：

- A2/A3 未选择；
- B16 不晋级；
- S0 保持默认；
- Graph、Dynamic、deferred、TP/EP/PP 和长上下文扩展保持未开启。
