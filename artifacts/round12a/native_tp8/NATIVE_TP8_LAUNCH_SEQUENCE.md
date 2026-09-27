# Round12a Native SGLang TP8 / old-QSA 启动顺序

本工作流不得与物理 12 上的 KT/non-KT 性能窗口重叠。物理 0–7 的 PCI `numa_node` 均为 `-1`，模型与 PLE checkpoint 还共享 Host page cache、存储和内存 cgroup 压力；当前没有证据支持并跑后仍保持公平计时。

1. 等待 v173–v177 的当前 Round12a 进程全部退出，核对 guard ACK、容器移除和物理 12 回收。
2. 完成当前实验块的原始归档、SHA/全文件校验，然后运行既有 KT checkpoint cache-retirement hook。缓存退休必须位于哈希之后。
3. 从远端只读父根 `/mnt/data1/kt-round9/round12-p2-baseline-v153` 分别构建 `/mnt/data1/kt-round9/round12a-native-tp8-v180` 至 `v183`；每个 root 先跑 host preflight，再在镜像 `601e76d17e9f` 内跑无设备 preflight。
4. 每个版本都新建两轮 60 秒 admission；四个版本依次为 v180 A、v181 B、v182 B、v183 A。每个进程只有 `warmup1 + measurement3`，请求固定为 64 input / 2 output、temperature=0、seed=20260920，自动重试为 0。
5. 每次启动前要求物理 0–7 无设备进程、逐卡 HBM 基线不超过 8192 MiB、NUMA2/3/4/7 和系统内存通过冻结门槛，并确认没有其他 KT 性能容器。
6. 每次进程结束后等待八卡 guard 的 OWNED_EXITED ACK，确认物理 0–7 进程表为空，再归档原始 evidence 并完成 SHA。之后才允许做 checkpoint cache retirement 和下一次 fresh admission。
7. v180–v183 全部完成后运行 `compare_native_tp8.py`。A/B 标签只表示相同实现的独立重复块，不表示两个实现；比较器检查 exact quality、old-QSA/rank/layer 命中、三次 measurement 中位数、逐卡 HBM 和 Host cgroup 峰值一致性。

任何 source/hash、TP reduction、old-QSA、输出、进程归属或资源门失败都 fail closed；不改用 promoted QSA，不降低门槛，也不自动换卡或并跑。

## 四进程角色与“ABBA”含义

v180/v183 是相同 Native TP8/old-QSA 构建的 repeat cohort A1/A2，v181/v182 是同一构建的 repeat cohort B1/B2。四个进程均为 Native TP8；A/B 只冻结 A/B/B/A 的启动顺序，用于检查独立进程间质量、时间和资源稳定性。它不构成 KT 与 Native 的实现对照，也不计算加速比。当前 KT/non-KT 主线使用 promoted QSA，而本工作流按用户要求固定 old QSA；不同 QSA 的结果不得称为公平 KT/Native A/B。

## 远端构建与无设备同镜像预检

工具目录只需传输 `artifacts/round12a/native_tp8` 的冻结 bundle，正式构建使用：

```bash
python3 native_tp8_builder.py \
  --parent /mnt/data1/kt-round9/round12-p2-baseline-v153 \
  --output /mnt/data1/kt-round9/round12a-native-tp8-v180 \
  --inputs /mnt/data1/kt-round9/round12a-native-tp8-tools \
  --version 180 \
  --queue /mnt/data1/kt-round9/round12a-native-tp8-tools/frozen-inputs/ROUND12A_ABBA_QUEUE.json
```

v181–v183 只替换版本和输出根。每个 root 先运行 host `native_tp8_offline_preflight.py`。随后使用镜像 `601e76d17e9f`、`--network none`、只读挂载 snapshot 和 Ascend driver，**不传任何 `--device`**，运行 snapshot 内的 `native_tp8_image_preflight.py`。该步骤只解析 ServerArgs 并导入冻结 provider/source tuple；必须记录 `physical_devices_exposed=[]`，不得初始化模型或 NPU context。
