# Round15 证据与复现指南

## 1. 证据层级

Round15 的证据分成四层：

1. **机器可读结论**：最终状态、配置、组件和请求比较 JSON/CSV。
2. **可重放工具**：snapshot builder、preflight、controller、driver、supervisor、probe 和 comparator。
3. **原始压缩证据**：包含请求结果、monitor、graph event、admission、source 和 supervisor 的 tar。
4. **身份与完整性**：build manifest、source SHA、archive sidecar 和总哈希清单。

Markdown 报告只解释这些证据，不替代它们。

## 2. 主要结论文件

| 文件 | 用途 |
|---|---|
| [FINAL_STATUS.json](../../../artifacts/round15/FINAL_STATUS.json) | 最终状态、范围、质量、资源和限制 |
| [FINAL_CONFIG.json](../../../artifacts/round15/FINAL_CONFIG.json) | 冻结后的默认配置；A/B 均 OFF |
| [P0_PROFILE_RESULT.json](../../../artifacts/round15/P0_PROFILE_RESULT.json) | 两个 fresh profile 块和热点数据 |
| [A1_COMPONENT_RESULT.json](../../../artifacts/round15/A1_COMPONENT_RESULT.json) | A 的两块组件 gate |
| [A_REQUEST_DIRECTIONAL_RESULT.json](../../../artifacts/round15/A_REQUEST_DIRECTIONAL_RESULT.json) | v341/v342 请求比较 |
| [B_COMPONENT_RESULT.json](../../../artifacts/round15/B_COMPONENT_RESULT.json) | v347/v357 组件证据与生产 int32 合同 |
| [B_REQUEST_DIRECTIONAL_RESULT.json](../../../artifacts/round15/B_REQUEST_DIRECTIONAL_RESULT.json) | v354/v360 请求比较和 graph 命中 |
| [BENCHMARK.csv](../../../artifacts/round15/BENCHMARK.csv) | A/B 统一绝对性能表 |
| [ROUND15_EXECUTION_BUDGET.json](../../../artifacts/round15/ROUND15_EXECUTION_BUDGET.json) | 全部 run 和预算 |
| [HASHES.json](../../../artifacts/round15/HASHES.json) | 31 项关键交付 SHA-256 |

## 3. 关键原始归档

| Run | 归档 | SHA-256 |
|---:|---|---|
| 357 | `b-graph-results/v357/20260928-round15-b-v357-archive.tar` | `52f8cc9e1df499e13433884f8e7d1617c886a78fd5cb145c22bfb67adcd6ced5` |
| 354 | `b-model-results/v354/evidence.tar` | `58128d7679f2e7410ea9cd428083ef810e8c1b2d4c7fd096f91df92c20958342` |
| 356 | `b-model-results/v356-nohit/evidence.tar` | `eac9b108b6c7fa7b05426b578fffbc877b35af29b6ed8c552e2f1c12e3a2b45d` |
| 358 | `b-model-results/v358-capture-selfcheck-failure/20260928-round15-b-v358-failure-archive.tar` | `a39b43fba2acda8387dfa9b4cf340c2073ee1f010babf17c4b689b30c450dbbb` |
| 360 | `b-model-results/v360/20260928-round15-b-v360-archive.tar` | `ab3914ba57e44825fe778427a04f2218642c7b72d0f11a3bc1400c4a700e5ac8` |

v341/v342 的 A 请求归档分别位于 `a-model-results/v341/evidence.tar` 和 `v342/evidence.tar`。P0 v335/v338、A 组件 v339/v340 和 admission tar 也在同一 artifacts 目录中。

## 4. 离线复算

### A 请求比较

先从 v341/v342 tar 中取出 `round15-a-model-final.json` 和 `monitor.jsonl`，再执行：

```bash
python3 artifacts/round15/compare_round15_a_model.py \
  --baseline <v341>/round15-a-model-final.json \
  --candidate <v342>/round15-a-model-final.json \
  --baseline-monitor <v341>/monitor.jsonl \
  --candidate-monitor <v342>/monitor.jsonl \
  --output /tmp/round15-a-result.json
```

输出应与 `A_REQUEST_DIRECTIONAL_RESULT.json` 字节一致，并检查 14 个请求身份、output IDs、两次 measurement 和实际 candidate binding。

### B 请求比较

```bash
python3 artifacts/round15/compare_round15_b_directional.py \
  --baseline artifacts/round15/b-model-results/v354/evidence.tar \
  --baseline-root 20260928-round15-b-graph-screen-baseline-s-v354 \
  --candidate artifacts/round15/b-model-results/v360/20260928-round15-b-v360-archive.tar \
  --candidate-root evidence/20260928-round15-b-graph-screen-candidate-s-v360 \
  --output /tmp/round15-b-result.json \
  --csv /tmp/round15-b.csv
```

comparator 会验证请求调度身份、output token exact、两次 W1/W2 measurement、capture/replay layer 和 marker、HBM/Host 监控以及 startup 时间。正常结果状态为 `ROUND15_B_DIRECTIONAL_SMALL_GAIN_BELOW_PROMOTION_GATE`。

### P0 和 A 组件

- `compare_round15_p0.py` 比较 v335 salvage 和 v338 process-PASS 的身份与 profile。
- `compare_round15_a1_component.py` 要求两个组件块所有 24 行都通过 15% parent gate、selected-set exact 和 downstream 数值门。

## 5. 源码和构建身份

主要源码：

- [round15_a_qsa_candidate.py](../../../artifacts/round15/round15_a_qsa_candidate.py)
- [round15_b_segment_policy.py](../../../artifacts/round15/round15_b_segment_policy.py)
- [round15_b_resident_graph.py](../../../artifacts/round15/round15_b_resident_graph.py)
- [build_round15_a_model_snapshot.py](../../../artifacts/round15/build_round15_a_model_snapshot.py)
- [build_round15_b_model_snapshot.py](../../../artifacts/round15/build_round15_b_model_snapshot.py)

每个运行 snapshot 有独立 build/preflight JSON。v360 manifest SHA-256 为 `289807c45b7bac8f21bf4f108f367d15fb9758831cae7d534d1fe847d20173ae`，binding SHA-256 为 `be3a838819c7bbe4076de841fbf397568071b2dbc117d2e492a62c629943aa6f`。

## 6. 哈希验证

下面的离线脚本可验证 `HASHES.json` 中所有条目：

```bash
python3 - <<'PY'
import hashlib, json
from pathlib import Path

manifest = json.loads(Path('artifacts/round15/HASHES.json').read_text())
for entry in manifest['entries']:
    path = Path(entry['path'])
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == entry['sha256'], (path, digest, entry['sha256'])
    assert path.stat().st_size == entry['size_bytes']
print(f"PASS: {len(manifest['entries'])} Round15 artifacts")
PY
```

对大 tar 可使用流式 `sha256sum`，不需要解包。不要在共享 A3 上为了验证文档重新加载模型；原始归档和 comparator 足以复算本轮结论。

## 7. 复现边界

- 完整设备复跑仍必须执行 fresh 两轮 60 秒 admission，不能依据历史空闲状态直接启动。
- donor torch_npu 2.10 API 不能替代本地 torch_npu 2.9 实际能力。
- 请求比较的两进程简化口径是用户批准的最终范围；复算能证明原结果一致，不能把它提升成四进程 ABBA。
- `HASHES.json` 只覆盖最终冻结的 31 项关键文件；未列出的重复快照和解包副本不是发布身份的一部分。
