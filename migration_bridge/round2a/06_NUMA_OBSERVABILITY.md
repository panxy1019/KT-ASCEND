# NUMA / Affinity Observability

## Status

`PASS (observability)` — A3 的 8 个 NUMA node、进程 CPU affinity、CPUInfer subpool 数量、subpool→node 和 threads/pool 均可从只读 runtime API 获取。

## Data path

```text
BaseMoEWrapper(numa_nodes, threadpool_count, cpuinfer_threads)
  -> WorkerPoolConfig.subpool_count
  -> WorkerPoolConfig.subpool_numa_map
  -> WorkerPoolConfig.subpool_thread_count
  -> CPUInfer
  -> CPUInfer.worker_pool_config()
  -> BaseMoEWrapper.cpu_runtime_diagnostics()
```

source：

- config construction/cache：`kt-kernel/python/experts_base.py:BaseMoEWrapper._get_cpu_infer:175-211`
- native read-only accessor：`kt-kernel/ext_bindings.cpp:PYBIND11_MODULE:550-563`
- affinity/node enumeration：`kt-kernel/python/experts_base.py:BaseMoEWrapper.cpu_runtime_diagnostics:213-236`

## A3 observed values

```text
available_numa_nodes = [0,1,2,3,4,5,6,7]
process_cpu_affinity = [0,1,2,3,4,5,6,7]

case 1:
  subpool_count        = 1
  subpool_numa_map     = [0]
  subpool_thread_count = [4]

case 2:
  subpool_count        = 2
  subpool_numa_map     = [0,1]
  subpool_thread_count = [2,2]
```

这里的 CPU affinity `[0-7]` 是 disposable container 的 cpuset；node visibility 来自 `/sys/devices/system/node/online`。测试验证请求的 map 与 CPUInfer backend 中的实际 config 相同，而不是只回显 Python 构造参数。

## Container limitation

非特权容器中 hwloc/libnuma 输出 `set_mempolicy`/memory-bind permission errors。故本轮结论严格限定为：

- worker pool mapping 可观测：是；
- CPU calculation correctness：是；
- process affinity 可观测：是；
- NUMA nodes 可见：是；
- OS 实际 memory placement/membind 生效：**未验证**。

这不阻塞 Round 2A 的 observability gate；不能把它表述为 NUMA placement 或性能已验证。

## Explicit non-goals

没有实现自动 node 选择、topology-aware expert placement、跨 NUMA migration、performance tuning 或 CPU/NPU affinity mapping。
