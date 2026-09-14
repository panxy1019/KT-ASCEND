# Current KTransformers Architecture（Round 1）

状态：`CODE_INSPECTED`。除 E003 中单独标注的 TaskQueue 测试外，本文件描述源码结构，不表示完整扩展已在 A3 运行。

## Top-level path

```text
Python / SGLang
  -> KTMoEWrapper factory
  -> BaseMoEWrapper / backend-specific wrapper
  -> kt_kernel_ext.CPUInfer
  -> TaskQueue
  -> WorkerPool (NUMA subpools)
  -> backend MoE forward_task
  -> CPU result buffer
```

关键入口：

- `kt-kernel/python/experts.py:33-49` 定义 inference method 集合：AMXINT4/8、RAWINT4、FP8、BF16、FP8_PERCHANNEL、GPTQ_INT4、SYCL_GPTQ_INT4、MXFP4/8、LLAMAFILE、MOE_INT4/8。
- `kt-kernel/python/experts.py:72` 是公共 factory `KTMoEWrapper`。
- `kt-kernel/python/experts.py:316-360` 的实际 mapping：AMX → `AMXMoEWrapper`；native formats → `NativeMoEWrapper`；LLAMAFILE → `LlamafileMoEWrapper`；MOE_INT4/8 → `GeneralMoEWrapper`。
- `kt-kernel/python/experts_base.py:227` 是 inference 公共基类 `BaseMoEWrapper`。

## CPUInfer and WorkerPool

`BaseMoEWrapper` 通过类级 singleton 创建 CPUInfer：

- `experts_base.py:159-198`：首次调用创建 `WorkerPoolConfig` 和 `CPUInfer`。
- `experts_base.py:179-195`：显式 NUMA list 必须与 threadpool count 等长；否则采用 `[0..threadpool_count-1]`；总线程按余数均匀分配到 subpool。
- `cpu_backend/worker_pool.h:132-136`：配置字段为 `subpool_count`、`subpool_numa_map`、`subpool_thread_count`。
- `cpu_backend/worker_pool.cpp:414-443`：每个 subpool 在线程中先 `set_to_numa`，再创建对应 `InNumaPool`，最后建立 `NumaJobDistributor`。
- `ext_bindings.cpp:548-564`：pybind 导出 `WorkerPoolConfig`、`CPUInfer`、`submit`/`sync`；非 `KTRANSFORMERS_CPU_ONLY` 时才导出 stream-aware API。

CPUInfer 本体：

- `cpu_backend/cpuinfer.h:38-62`：支持总线程数、线程数+单 NUMA、完整 `WorkerPoolConfig` 三种构造。
- `cpuinfer.h:80-85`：`submit` 将 CPUInfer 写入 task 参数并调用 task thunk。
- `cpuinfer.h:87-95`：`submit_with_cuda_stream` 通过 vendor alias `cudaLaunchHostFunc` 把 task 放入设备 stream callback。
- `cpuinfer.h:98-118`：`sync`/`sync_with_cuda_stream` 最终调用 TaskQueue 的 `sync(allow_n_pending)`。
- `task_queue.cpp:42-50`：enqueue 增加 pending 并通知单独的 queue worker。
- `task_queue.cpp:53-64`：等待 `pending <= allow_n_pending`，并向调用者重抛第一个 task exception。

代码审计风险：`cpuinfer.h:109` 和 `:116` 分配 `SyncArgs`，`sync_`（`:103-106`）未释放。这里只记录可见事实，不在 Round 1 做推测性修复。

## KExpertsCPUBuffer

`kt-kernel/python/experts_base.py:75-142` 管理双深度缓存：

| Buffer | dtype/location | 证据 |
|---|---|---|
| input | BF16 CPU pinned | `:100-103` |
| immediate expert IDs | int64 CPU pinned | `:104-107` |
| deferred expert IDs | int64 CPU pinned，初值 -1 | `:108-111` |
| routing weights | FP32 CPU pinned | `:112-115` |
| CPU output | BF16 CPU pinned | `:116-119` |
| batch-size scalar | int32 CPU pinned | `:120-123` |
| device output | 与输入 device/dtype 一致 | `:124-127` |

- `buffer_depth = 2`：`experts_base.py:86`。
- 当前槽为 `layer_idx % 2`，下一槽循环递增：`:406-407`。
- `:422-438` 拷入输入、权重、immediate IDs，并异步提交 forward task。
- `:440-455` 可选提交 deferred task；其结果写到 next slot，并记录 layer pending 状态。
- `:479-483` 根据该层是否有 deferred task，将 `allow_pending` 设为 1 或 0，stream-sync 后把 CPU output 拷回 device output。

这套设计把相邻层的 deferred task 作为允许保留的一个 pending 项。它依赖 layer index、双 buffer、单例 CPUInfer 和调用顺序共同保持一致。

## MOE_INT8 path

```text
KTMoEWrapper(method="MOE_INT8")
  -> experts.py:_create_inference_wrapper
  -> GeneralMoEWrapper
  -> kt_kernel_ext.moe.Int8_KERNEL_MOE
  -> MOE_KERNEL_TP<GemmKernelInt8>
  -> forward_task -> TaskQueue -> WorkerPool/subpools
```

- `python/utils/moe_kernel.py:11-24` 尝试导入 `Int8_KERNEL_MOE`/`Int4_KERNEL_MOE` 并记录能力。
- `moe_kernel.py:29-53` 定义 `GeneralMoEWrapper`。
- `moe_kernel.py:147-180` 的 online quantize/save 路径把 pool、尺寸、权重指针和 physical-to-logical map 交给 C++ task。
- `moe_kernel.py:275-312` 的 load 路径传递 gate/up/down weight 与 scale 指针，然后提交 `load_weights_task` 并同步。
- `ext_bindings.cpp:915-920`：启用 `USE_MOE_KERNEL` 时导出 Int8；Int4 额外要求 `__aarch64__ && CPU_USE_KML`。
- `operators/moe_kernel/la/kernel.hpp:296-343`：Int8 kernel 的元素类型为 int8、累加输出为 int32，并命名为 `MOE_INT8`。
- `kernel.hpp:345-371`：decode/prefill 使用不同的 up/gate 和 down N-block 配置，并强制维度可整除。

当前提交无法完成 KML 构建，因此 quantization layout、scale 解释及 decode/prefill 数值正确性没有得到 A3 运行验证；不能仅凭类型和字段名宣称兼容。

## Ascend vendor gap

`kt-kernel/cpu_backend/vendors/` 当前只包含：

```text
README.md  cuda.h  hip.h  maca.h  musa.h  vendor.h
```

对 `kt-kernel/cpu_backend`、CMake 与 setup 的静态搜索结果：

- 无 `ascend.h`。
- 无 `KTRANSFORMERS_USE_ASCEND`。
- 无 `aclrtLaunchHostFunc` 或 `aclrtMemcpyAsync`。
- 无 `torch_npu`。
- `cpuinfer.h:20-29` 只选择 CUDA/CUDA host callbacks、MUSA、ROCm/HIP、MACA vendor header。
- `cpuinfer.h:93,117` 的 stream callback 仍写成 CUDA-compatible API。

因此结论仅为：**当前 CPUInfer 的 device-stream bridge 没有源码级 Ascend vendor 分支**。本轮没有实现它，也没有验证能否通过兼容层工作。

