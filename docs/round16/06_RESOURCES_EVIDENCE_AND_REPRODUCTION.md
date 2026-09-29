# 资源、缓存退休、证据与复现

## 1. 正式资源身份

- 物理 NPU：12；
- `npu-smi`：card 6 / chip 0；
- NUMA：2/3/4/7；
- C128：4 × 32 workers；
- 正式请求并发：1；
- fresh admission：按用户更新采用 2 × 10 秒。

四个正式请求进程的最低平均 idle 高于 99.6%；最低 NUMA free memory 出现在 node4，约 120.29 GiB。每次启动前均重新核验设备进程、容器身份、训练 cgroup、HBM 和 Host/NUMA 门。

## 2. B 完整模型资源

B 日志中的模型初始化账本：

| 项目 | 数值 |
|---|---:|
| model weights | 24.2668 GB |
| peak activation | 2.85 GiB |
| KV cache | 23.04 GiB |
| KV tokens | 3,730 |
| torch reserved after warmup | 47.34 GiB |
| torch allocated after warmup | 47.30 GiB |
| graph memory | 0 GiB |

B wrapper 没有冻结 Host cgroup peak，这一字段保留为 UNKNOWN。没有用容器 640 GiB 上限代替真实峰值。

## 3. 定向 page-cache 退休

每个正式实验块都遵守：

```text
自有模型进程退出
  -> 容器/guard 结束
  -> 原始结果归档并计算 SHA
  -> 全机 FD/mmap 检查登记文件无人引用
  -> 仅对登记的只读模型与 bank 文件调用 POSIX_FADV_DONTNEED
  -> fincore 复核
  -> NUMA MemFree / system MemAvailable 回执
```

登记集合为 131 个 checkpoint 文件和 48 个 bank 文件，共 179 个文件。

| Block | 退休前 resident bytes | 退休后 resident bytes |
|---|---:|---:|
| S0-A | 296,554,708,992 | 0 |
| B-A | 348,995,039,232 | 0 |
| B-B | 209,022,619,648 | 0 |
| S0-B | 293,700,280,320 | 0 |

没有执行全局 `drop_caches`，没有修改其他任务，也没有在 SHA/全文件校验之前退休缓存。

## 4. 最终资源状态

最终只读核验：

- Round16 自有容器：0；
- Round16 controller/model/guard：0；
- 物理12：`No process in device.`；
- 历史 timer：保持暂停。

证据：[FINAL_RESOURCE_CENSUS.txt](../../artifacts/round16/FINAL_RESOURCE_CENSUS.txt)。

## 5. GitHub 证据策略

GitHub 提交包括：

- 本轮实现源码和测试；
- 最终 Markdown 报告；
- A/B 权威分析 JSON；
- V1–V5 成功阶段 receipt；
- multi-page KV 结果；
- 四个正式请求块的 result、contract、container exit/remove 与 SHA；
- 2×10 admission 汇总与 cache-retirement receipt；
- benchmark、终态和最终哈希清单。

不提交：

- 模型 checkpoint 或 MXFP4 bank；
- 四个约 58–61 MB 的 A1 原始 JSONL trace；
- 可再生 monitor/server 大日志；
- 第三方 `py-spy` wheel/二进制；
- 容器内复制的完整 vLLM/vLLM-Ascend donor 源码镜像；
- Python `__pycache__`；
- 凭据或 SSH 密码。

这些大体积原始证据仍保留在本地和集群原 evidence 目录。最终 SHA 清单记录了本轮冻结证据的身份；GitHub 上的发布清单另列实际纳入提交的文件。

发布内容逐文件 SHA-256、字节数和排除项说明见 [GITHUB_PUBLISH_CONTENTS.json](../../artifacts/round16/GITHUB_PUBLISH_CONTENTS.json)。该清单本身的哈希保存在同目录的 `GITHUB_PUBLISH_CONTENTS.sha256`。

## 6. 验证

最终验证：

- 327/327 unit tests PASS；
- `compileall` PASS；
- 310 个 JSON 文件解析 PASS；
- benchmark 12 行检查 PASS；
- KV scope regression PASS；
- `git diff --check` PASS；
- 最终冻结清单 238 个文件；
- manifest SHA-256：`5136ca9a2d96f0c5bc76ca8fc6e2b5662a3edc26c5b7dd8d1574560d91a46dcc`。

权威回执：[ROUND16_FINAL_VERIFICATION.json](../../artifacts/round16/ROUND16_FINAL_VERIFICATION.json)。

## 7. 最小复现命令

CPU-only 单元测试：

```bash
cd /path/to/KT-ASCEND
python3 -m unittest discover -s tests/round16 -p 'test_*.py'
python3 -m compileall -q round16 round16_integration tools tests/round16
git diff --check
```

解析最终机器产物：

```bash
python3 - <<'PY'
import csv, json
from pathlib import Path

status = json.loads(Path('artifacts/round16/FINAL_STATUS.json').read_text())
rows = list(csv.DictReader(Path('artifacts/round16/ROUND16_BENCHMARK.csv').open()))
assert status['status'] == 'ROUND16_PARTIAL_WITH_GAPS'
assert status['b']['disposition'] == 'B16_MEASURED_NO_PROMOTION'
assert len(rows) == 12
print(status['default_runtime'])
PY
```

真实 NPU 重跑仍需冻结模型、bank、镜像、resident map、物理12和 C128 资源合同；这些大资产不会从 GitHub 自动下载或重建。
