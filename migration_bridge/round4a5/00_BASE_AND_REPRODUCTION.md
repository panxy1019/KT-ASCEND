# Round 4A.5 P2 干净环境复现基线

状态：`P2_SAME_PATH_NONDETERMINISM_REPRODUCED`

日期：2026-09-03

## 1. 冻结代码与数据

- 父仓库分支：`feature/kt-round4a5-p2-determinism`
- 父仓库提交：`2ae5523ce1d1b30f60baf78917af1229f57a34aa`
- SGLang 子仓库提交：`cbaa79c2f5b004ab4ca470c9fa56161666ccafb2`
- 模型：DeepSeek-V2-Lite revision `604d5664`
- P2 placement：CPU-enabled layers `{1, 9, 17, 26}`，每层 4 个 CPU experts
- P2 GGUF size：`8,858,371,392` bytes
- P2 GGUF SHA256：`2c4cd307a53b761c4191cf108d1e00ec2bd2c99b48a595a28ac8fdd0e0edd1fa`

## 2. 新实验容器

- 容器：`kt-r4a5-p2-debug`
- 镜像：`quay.io/ascend/verl:v0.8.0-cann9.0.0-torch_npu2.9.0.post2-a3-ubuntu22.04-py3.11-vllm`
- NPU：仅 `/dev/davinci0`
- CPU affinity：`0-15`
- shared memory：16 GiB
- network：host
- 源码挂载：`/home/admin/projects/KT-ASCEND-round4a5 -> /workspace/kt-src`
- 结果挂载：`/home/admin/kt-artifacts/round4a5 -> /workspace/results`
- P2 GGUF 挂载：只读
- 模型、Ascend driver 和 firmware：只读

旧容器 `kt-r4a4-pairwise` 保持停止且未修改。

## 3. 容器内依赖与构建

基础镜像缺少旧实验容器曾安装的构建/运行依赖。本轮只在 disposable
container 内补齐：

- `pkg-config`
- `libhwloc-dev`
- `libnuma-dev`
- `IPython==9.16.1`
- `psutil==7.2.2`
- 从旧容器复制 `sgl_kernel_npu==2026.6.1`

未修改 A3 宿主系统包。后续依赖禁止直接从境外源下载，优先复用旧容器或使用
已配置的国内镜像。

kt-kernel 在新工作树中重新构建成功，Ascend runtime bridge、LLAMAFILE ARM
路径和 NUMA support 均启用。构建日志保存在 A3：

```text
/home/admin/kt-artifacts/round4a5/r4a5-build-ext.log
/home/admin/kt-artifacts/round4a5/r4a5-build-py.log
```

## 4. 冻结运行参数

- TP=1，batch=1，BF16，greedy，seed=0
- CPUInfer threads=16，threadpool count=1，NUMA node=0
- Graph、Deferred Experts、Dynamic Placement、MTP、Speculative OFF
- NPU0 only
- P2 placement 和 numerical contract 不变

启动入口仍使用：

```text
migration_bridge/round4a4/tools/launch_quality_placement.sh p2
```

## 5. 最小复现结果

同一个新 serving process 中，对两个最短历史失败 prompt 分别运行 10 次
64-token greedy generation，并检查 8/16/32-token prefix：

| Prompt | 64-token unique hashes | repeat exact | prefix exact |
|---|---:|---:|---:|
| `v_en_01` | 9 / 10 | FAIL | FAIL |
| `v_struct_01` | 10 / 10 | FAIL | FAIL |

结果：

```text
all_repeat_exact = false
all_prefix_exact = false
P2_SAME_PATH_NONDETERMINISM = REPRODUCED
```

证据文件 SHA256：

```text
7a4e74ce97b6eab803b084b9eb6259fea33c8f9023cf30aebf80c0e80fe54ef8
```

这证明 P2 blocker 不依赖旧容器 writable-layer 状态。下一步可以进入 matched-history
逐层归因，而不是继续处理环境漂移。

## 6. 当前停止状态

复现结束后已先终止 SGLang scheduler 和 server，端口 31000 已释放，NPU0 无运行
进程。新容器保留运行，仅 PID 1 的 `sleep infinity` 存活，便于继续调试。

