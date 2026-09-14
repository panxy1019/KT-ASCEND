# E004 — MOE_INT8/KML Single-layer Reference

状态：`BLOCKED / NOT RUN`

## Planned comparison

按任务要求，目标是在 CPU-only、无模型、无 SGLang、无 graph、无 NPU 条件下比较一个最小 Torch reference MoE 与 `MOE_INT8/KML`，记录：

- max absolute error
- mean absolute error
- relative L2 error

## Blocking chain

```text
E002 CMake configure failure
  -> no kt_kernel_ext
  -> no Int8_KERNEL_MOE import
  -> no MOEConfig/load_weights_task/forward_task execution
  -> no comparable KML output
```

此外，当前 commit 的 KML mat-kernel source directories 不在 Git tree，现有 KML bench 又使用已经不匹配的 pybind symbol 名。为避免用假实现或不同版本 wheel 产生误导性数字，本轮没有构造 numerical result。

## Result fields

| Metric | Result |
|---|---|
| max_abs_error | N/A — kernel not built |
| mean_abs_error | N/A — kernel not built |
| relative_l2 | N/A — kernel not built |
| illegal instruction | not observed; execution not reached |
| numerical correctness | unknown |

这满足 Round 1 的“失败时准确定位、不要直接大改”边界；不能将 `BLOCKED` 表述成 numerical FAIL。

