# Round18 EP1 独立 expert-list consumer 与图内存突变探针合同

日期：2026-09-30
状态：`OFFLINE_READY_DEVICE_NOT_RUN`
范围：单层、TP1/EP1、冻结 `AscendMoERunner`、组件级手工 NPUGraph。

## 目的

该探针验证上一份源码审计得到的最小可行假设：冻结 Ascend BF16 通用 MoE 路径已经能消费 `w13_weight_list` / `w2_weight_list`，因此可以直接为每个 physical expert 创建独立 NPU storage，而无需先创建 Q32 stacked parent、`unbind().clone()` 或在每次 miss 时重建 parent。

本探针不替换 `AscendMoERunner`。runner 继续在 512 logical expert 域执行原生 Top-10 与 AllGather consumer；量化 method 在 runner 构造后按既有 Round18 seam 重新绑定到 `logical512/global512/local32/EP1` 的 physical config。设备权重为 32 对独立 tensor：

- `w13_weight_list[i]`: BF16 `[2560, 1280]`；
- `w2_weight_list[i]`: BF16 `[640, 2560]`；
- 每个 tensor 有独立 storage，不能是 stacked parent 的 view；
- slot 地址在 A→B→A 全周期保持稳定。

## 发布顺序

固定 slot 替换严格遵守：

1. 将一个 logical expert 的 checkpoint 行在 CPU 侧转换为 native consumer orientation；
2. 原位写 `w13` 独立 destination；
3. 原位写 `w2` 独立 destination；
4. 完成 ready barrier；
5. 才原位更新同一个 512-entry device `expert_map`；
6. generation 单调递增。

第二 projection 写失败时，旧 map 仍然是唯一 published map。部分写入的 slot 不可达，直到一次完整重试成功。复制通过 tensor `copy_` 完成，不使用 raw pointer。

## eager 与图验证

eager 必须先通过冻结 native consumer：

- A（原 resident）输出；
- slot0 A→B，B 成为 resident；
- B 输出；
- slot0 B→A；
- A 输出恢复，hash 与初始 A 一致；
- 三次均对独立 BF16 reference 满足冻结误差门。

之后只尝试手工组件 NPUGraph：

1. capture A，但 capture output 明确不可消费；
2. 第一次合法 replay A 后才能检查输出；
3. replay 完成边界外执行 A→B 和静态 route-input 更新；
4. replay B，验证图观察到新 weight bytes 与同地址新 map 值；
5. replay 完成边界外恢复 B→A；
6. replay A，验证恢复。

该证据即使通过也只能写 `COMPONENT_NPUGRAPH_ONLY`，不能写 full-model graph、FULL wrapper 或请求性能通过。冻结 wrapper 太重或手工 NPUGraph 不支持该 consumer 时，eager 结果仍独立保存为 `PASS_EP1_INDEPENDENT_LIST_EAGER_GRAPH_BLOCKED`。

## 实现与离线验证

- 合同：`round18/independent_expert_list.py`
- 设备入口：`round18_integration/native_independent_expert_list_probe.py`
- 合同测试：`tests/round18/test_independent_expert_list.py`
- 入口测试：`tests/round18/test_native_independent_expert_list_probe.py`

离线命令：

```bash
python3 -m unittest \
  tests.round18.test_independent_expert_list \
  tests.round18.test_native_independent_expert_list_probe
```

设备入口有双重门：CLI `--enable-device` 与环境变量 `ROUND18_ENABLE_EP1_LIST_PROBE=1`。当前没有启动设备、远端进程或模型，设备状态保持 `NOT_RUN`。

## 设备放行后的预期命令形状

冻结镜像内提供实际路径：

```bash
ROUND18_ENABLE_EP1_LIST_PROBE=1 python -m \
  round18_integration.native_independent_expert_list_probe \
  --enable-device \
  --source-token SOURCE_PREFLIGHT.json \
  --snapshot-manifest S0_SNAPSHOT.json \
  --model-root MODEL_ROOT \
  --layer 0 \
  --output RECEIPT.json
```

设备执行前仍需使用父任务安排的资源 lease 和冻结镜像身份。此文档不授权设备启动。
