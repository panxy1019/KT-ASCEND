# E003 — CPUInfer Smoke

状态：`BLOCKED`（完整 CPUInfer）；`PASS`（仓库已有 TaskQueue 独立测试）

## Intended checks

任务要求验证：

```python
import kt_kernel
from kt_kernel import KTMoEWrapper
from kt_kernel import kt_kernel_ext
```

并检查 extension variant、`Int8_KERNEL_MOE`、`WorkerPoolConfig`、`CPUInfer` 构造、submit/sync 多轮无 crash。

## Why full smoke was not run

E002 在 CMake configure 阶段失败，未生成 `kt_kernel_ext`。因此 Python import、pybind symbol、WorkerPoolConfig 和 CPUInfer 构造均为 `BLOCKED`，不能标记为 FAIL-at-runtime，也不能用另一个旧 wheel 代替冻结提交。

## Safe lower-level test

仓库现有 `kt-kernel/cpu_backend/test/test_task_queue.cpp` 不依赖完整 extension/KML。将其与 `task_queue.cpp` 在 A3 隔离目录中编译运行，限制到 CPU 0–1：

```bash
g++ -std=c++20 -O0 -pthread \
  cpu_backend/task_queue.cpp \
  cpu_backend/test/test_task_queue.cpp \
  -o build_task_queue_test
taskset -c 0-1 ./build_task_queue_test
```

结果：`rc=0`。

## Evidence boundary

该结果只验证当前源码的 standalone TaskQueue test 在 A3 aarch64/GCC 11 环境通过。它不验证：

- CPUInfer/WorkerPoolConfig pybind；
- hwloc/NUMA worker pools；
- KML；
- stream host callbacks；
- MOE_INT8；
- 多轮 full CPUInfer 生命周期。

