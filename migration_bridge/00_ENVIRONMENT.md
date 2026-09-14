# B Round 1 Environment

审计日期：2026-08-26（UTC）

## 证据等级

- `B_VERIFIED`：在 B 机当前 checkout 上执行命令得到。
- `A3_VERIFIED`：通过 SSH 在 A3 上执行只读命令，或在隔离目录/既有空闲开发容器内执行受限构建得到。
- `CODE_INSPECTED`：只由当前冻结提交的源码静态审计得到，未声明运行成功。
- `BLOCKED`：前置构建失败，无法进入该验证阶段。

## Repository freeze（B_VERIFIED）

| 项目 | 值 |
|---|---|
| checkout | `/home/admin/Desktop/KT_ASCEND/ktransformers` |
| origin | `https://github.com/kvcache-ai/ktransformers.git` |
| branch | `main`，跟踪 `origin/main` |
| KTransformers commit | `c40d37c63cb9b7d041c8489167b3b822961d12c5` |
| commit time | `2026-08-25T01:48:41+08:00` |
| commit subject | `docs: add fine-tuning milestones to README updates (#2171)` |
| version | `0.7.0`（`version.py:6`） |
| SGLang commit | `0f36b26d7523351ec88b1e694e6d810234911c12` |
| SGLang ref description | detached HEAD；`remotes/origin/release/transformers-kt-post2` |
| initial worktree | clean |

冻结命令：

```bash
git remote -v
git branch --show-current
git status --short --branch
git rev-parse HEAD
git log -1 --format='%H%n%cI%n%s'
git submodule status --recursive
python3 -c 'exec(open("version.py").read()); print(__version__)'
```

审计开始时子模块未初始化。执行了仓库标准、可复现的初始化操作：

```bash
git submodule update --init --recursive
```

固定的主要子模块版本：

- `third_party/sglang`: `0f36b26d7523351ec88b1e694e6d810234911c12`
- `third_party/custom_flashinfer`: `fd94393fb5b8ba8bae9c0bd6ab1c2a429d81ac76`
- `third_party/llama.cpp`: `a94e6ff8774b7c9f950d9545baf0ce35e8d1ed2f`
- `third_party/pybind11`: `bb05e0810b87e74709d9f4c4545f1f57a1b386f5`

初始化后主仓库仍为 clean；Round 1 只新增本目录中的审计文档。

## A3 host（A3_VERIFIED）

连接目标为用户指定的 A3。本文不记录口令。除隔离构建目录外，命令均为只读；没有安装包、重启服务、停止进程、修改驱动/CANN、分配 NPU、运行模型或触碰集群作业。

| 项目 | 实测值 |
|---|---|
| hostname | `a3-server-00` |
| machine | Huawei Atlas 800T A3 |
| OS | openEuler 24.03 LTS-SP1 |
| kernel | `6.6.0-72.0.0.76.oe2403sp1.aarch64` |
| architecture | `aarch64` |
| CPU | 4 × Kunpeng 920 7280Z |
| CPU topology | 4 sockets，80 cores/socket，2 threads/core，640 logical CPUs，320 physical cores |
| NUMA | 8 nodes；每节点 80 logical CPUs（0-79、80-159、…、560-639） |
| memory | 2.0 TiB；审计时约 1.8 TiB available |
| `/home` | 3.4 TiB，总使用约 79%，约 719 GiB free |
| host Python | `python` 不存在；`python3 3.11.6` |
| host compiler | GCC/G++ 12.3.1 |
| host CMake | 不存在 |
| host PyTorch | 未安装 |
| host torch_npu | 未安装 |
| host hwloc/pkg-config | RPM 查询未安装 `hwloc`、`hwloc-devel`；`pkg-config` 不可用 |
| host numactl | `numactl`/`numactl-devel` 未安装；动态链接器仅发现 `libnuma.so.1` |

主机审计时负载约 37–38，存在大量活跃 VLLM worker。该事实是选择空闲开发容器、限制 CPU affinity 与并行度的依据，不代表这些负载由本轮产生。

## A3 NPU/software stack（A3_VERIFIED，未执行 NPU 计算）

| 项目 | 实测值 |
|---|---|
| NPU count | 16（ID 0–15） |
| generic product string | `Ascend910` |
| board/NPU name | `9382`，Chip Version V1 |
| HBM | 每卡报告 65536 MiB |
| driver package | `26.0.rc1` |
| ascendhal | `7.35.23` |
| firmware | `9.0.0.0.205`，package version `26.0.rc1` |
| host `ASCEND_HOME_PATH` | unset |
| host `ASCEND_TOOLKIT_HOME` | unset |
| host `LD_LIBRARY_PATH` | unset |

审计瞬时状态：NPU 1–15 被 VLLM 作业高占用；NPU 0 未列出运行 NPU 进程但已有约 3 GiB HBM 使用。Round 1 没有调用任何 NPU kernel，也没有据此声称 NPU 0 可安全使用。

## Isolated build environment（A3_VERIFIED）

为避免污染宿主环境，选择审计时处于空闲状态的既有开发容器 `verl-0.8.0-a3`，未改其镜像或安装依赖。

| 项目 | 值 |
|---|---|
| image | `quay.io/ascend/verl:v0.8.0-cann9.0.0-torch_npu2.9.0.post2-a3-ubuntu22.04-py3.11-vllm` |
| Python | 3.11.15 |
| GCC/G++ | 11.4.0 |
| CMake | 4.4.2（命令输出；容器 dpkg 同时有 3.22.1 元数据） |
| torch | `2.9.0+cpu` |
| torch_npu | `2.9.0.post2` |
| CANN | `9.0.0` |
| Ascend path | `/usr/local/Ascend/cann-9.0.0` |
| ACL header | 存在；`aclrtLaunchHostFunc` 可在 `acl_rt.h` 中找到 |
| KML | `/usr/local` 与 `/opt` 的受限搜索未找到 |
| pkg-config/hwloc | 此容器中不存在 |

隔离路径：`/home/admin/kt_ascend_round1_c40d37c`。创建前确认路径不存在，只拷贝了冻结提交的 `version.py`、`kt-kernel`、`third_party/llamafile`、`third_party/llama.cpp` 和 `third_party/pybind11`，约 100 MiB。构建使用 `taskset -c 0-3` 和并行度 4；没有覆盖 A3 上任何已有 checkout。

## Safety boundary

- 未执行 `apt`、`dnf`、`yum`、`pip install` 或系统配置写入。
- 未重启/停止容器和集群进程。
- 未修改 CANN、driver、firmware、环境模块或全局环境变量。
- 未运行 NPU 测试、模型、SGLang、HCCL、graph 或多卡任务。
- 唯一远端持久变化是上述新建隔离目录及其构建产物，保留用于精确复现。

