# B Round 1 SGLang KT Call Graph

状态：`CODE_INSPECTED`，未运行 SGLang、模型、CUDA 或 Ascend。

## Activation and construction

```text
CLI --kt-*
  -> ServerArgs dataclass / argparse
  -> ServerArgs validation
     -> kt_weight_path enables KT mode
     -> shared-expert fusion is disabled
  -> DeepSeek V4 model module side-effect imports kt_ep_wrapper
  -> kt_ep_wrapper registers "kt_ep" wrapper (priority 20)
  -> FusedMoE.__init__ selects base GPU quant method
  -> maybe_wrap_moe_quant_method(...)
  -> create_kt_config_from_server_args(layer_id)
  -> KTEPWrapperMethod(base_gpu_method, KTConfig)
  -> KTMoEWrapper(method=...)
  -> BaseMoEWrapper/backend
```

### CLI and ServerArgs

- `third_party/sglang/python/sglang/srt/server_args.py:723-737` 定义 KT fields，包括 weight path、method、CPUInfer threads、threadpool count、NUMA nodes、GPU expert 数量/比例、deferred experts、prefill threshold、dynamic update 和 placement。
- `server_args.py:4828-4905` 注册对应 CLI；默认 method 是 `AMXINT4`，默认 threadpool count 是 2。
- `server_args.py:2625-2630`：一旦 `kt_weight_path` 非空，自动禁用 shared-expert fusion，避免 shared experts 被 CPU offload。

### Registration scope

- `models/deepseek_v4.py:46-48` 通过 `_try_side_effect` 导入 `mxfp4_deepseek`、`kt_ep_wrapper` 等模块。
- `kt_ep_wrapper.py:4458-4477` 注册 `kt_ep`，priority 20。
- `quant_method_registry.py:31-68` 保存 wrapper 并按 priority 链式包装。
- `fused_moe_triton/layer.py:275-296` 先选 base GPU method，再调用 `maybe_wrap_moe_quant_method`。

重要边界：当前静态搜索只发现 DeepSeek V4 模型模块负责该 side-effect import。注册表注释也明确将其描述为 DSV4 plugin。因此不能把这条 activation path 泛化成所有 SGLang MoE 模型都会自动启用 KT。

### Per-layer KTConfig

`kt_ep_wrapper.py:3149-3203`：

1. draft/speculative 等场景可由 `is_kt_ep_wrapper_disabled()` 跳过。
2. `kt_weight_path is None` 时直接返回 None，不包装。
3. 初始化全模型 GPU-expert masks，按当前 layer 取 mask。
4. 把 CPUInfer、threadpool、NUMA、weight、method、deferred/prefill/dynamic 配置写入 `KTConfig`。

`kt_ep_wrapper.py:3206-3229`：CPU expert ID 变成 `-1`，GPU expert ID 从 logical ID remap 到紧凑 GPU weight index。

`kt_ep_wrapper.py:3688-3730`：在 TP rank 0 构造 `KTMoEWrapper`；普通 inference 路径直接传入 method 与 deferred 配置，LoRA/SFT 使用另一分支。

## Forward call graph

```text
FusedMoE.forward / dispatcher
  -> KTEPWrapperMethod.apply
     1. staging_buffer.copy_(hidden_states)
     2. CPU stream waits for main stream
     3. KT wrapper submit_forward (CPU task, nonblocking)
     4. mask CPU IDs to -1; remap GPU IDs
     5. gpu_method.apply(masked dispatch)
     6. KT wrapper sync_forward on CPU stream
     7. main stream waits for sync event
     8. output = gpu_output + cpu_output
```

源码位置：

- apply 入口和语义：`kt_ep_wrapper.py:3958-3975`。
- staging 与 CPU submit：`:4150-4172`。
- mask/remap：`:4178-4189`。
- GPU expert compute 或 zero-GPU bypass：`:4195-4242`。
- CPU sync、event wait、结果合并：`:4248-4267`。
- wrapper 自身的 submit-compute-sync 设计说明：`:3481-3497`。

## CUDA assumptions relevant to future Ascend work

这是缺口清单，不是实现建议：

- CPU stream 创建/等待/上下文使用 `torch.cuda`（`:4165-4167`）。
- 权重加载前直接 `torch.cuda.synchronize()`（`:3746-3749`）。
- timing 与 event wait 使用 `torch.cuda.synchronize/current_stream`（`:4173-4176`, `:4264-4266`）。
- 下层 `BaseMoEWrapper` 调 `CPUInfer.submit_with_cuda_stream` / `sync_with_cuda_stream`，而 CPUInfer 无 Ascend vendor branch。

因此最新版 SGLang KT 路径的 CPU/GPU 并行结构已明确，但当前代码仍是 CUDA stream contract，Round 1 没有验证 torch_npu stream、ACL host callback、设备内存拷贝或 graph 语义。

