# B Round 1.5 — CPUInfer Smoke

## Result

**PASS / A3_VERIFIED**

The KML-off CPU-only extension imports, constructs `WorkerPoolConfig` and `CPUInfer`, submits work, synchronizes, and completes 1,000 routed-expert forwards without a crash.

## Import smoke

The staged CPU-only package was imported with an explicit `PYTHONPATH`. Observed symbols:

| Check | Result |
|---|---|
| `import kt_kernel` | PASS, version `0.7.0` |
| `import kt_kernel_ext` | PASS |
| `kt_kernel_ext.CPUInfer` | present |
| `kt_kernel_ext.WorkerPoolConfig` | present |
| `kt_kernel_ext.moe` | present |
| `kt_kernel_ext.moe.MOEConfig` | present |
| `kt_kernel_ext.moe.MOE` | present |
| `Int8_KERNEL_MOE` | absent, expected because `KTRANSFORMERS_CPU_MOE_KERNEL=OFF` |

Evidence: A3 `logs/r15/import-smoke.log`.

## Runtime configuration

```text
subpool_count       = 1
subpool_numa_map    = [0]
subpool_thread_count= [4]
iterations          = 1000
```

Each iteration performs:

```text
CPUInfer.submit(moe.forward_task(...))
CPUInfer.sync()
```

The test first submits and synchronizes `moe.load_weights_task()`.

## Workload

The smoke uses the real generic `kt_kernel_ext.moe.MOE` binding and LLAMAFILE implementation, not a synthetic TaskQueue-only task:

| Parameter | Value |
|---|---:|
| batch | 1 |
| hidden size | 32 |
| intermediate size | 256 |
| experts | 2 |
| top-k | 1 |
| selected expert | 1 |
| gate/up/down weights | deterministic in-memory F32 |
| input/output | BF16 |

The intermediate dimension is 256 because this path requires the LLAMAFILE `QK_K` alignment.

## Observed result

```text
iterations:                   1000
elapsed_seconds:              0.024799656
microseconds_per_iteration:   24.7997
all_finite:                   true
max_abs_error:                0.0
mean_abs_error:               0.0
relative_l2:                  0.0
```

The timing is retained only as smoke evidence. The tensor is intentionally tiny, so it is not a performance benchmark and must not be extrapolated to model throughput.

## NUMA limitation

The container emitted:

```text
set_mempolicy: Operation not permitted
hwloc_set_membind_nodeset: Operation not permitted
```

The computation and CPUInfer synchronization still completed correctly. Because the disposable container lacks the capability needed to change memory policy, this run verifies the worker pool and CPU execution, but **does not verify NUMA memory binding**. It also does not justify changing the host or granting broader container privileges.

## Reproduction asset

The exact test is retained at `migration_bridge/experiments/r15_cpu_llamafile_smoke.py`. Runtime output is in A3 `logs/r15/cpuinfer-llamafile-smoke.log`.

## Verified boundary

Verified: construction, load task, submit, sync, 1,000 forwards, finite output, deterministic numerical reference.

Not verified: concurrent callers, multi-subpool scheduling, NUMA memory placement, model-sized experts, quantized GGUF weights, whole-model integration, accelerator/CPU overlap, or performance.
