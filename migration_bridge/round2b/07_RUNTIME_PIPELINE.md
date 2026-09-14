# End-to-End Runtime Pipeline

## Executed path

```text
NPU BF16/int64/float32 inputs
  -> asynchronous D2H into pinned tensors
  -> CANN host callback submits real LLAMAFILE MoE task
  -> independent NPU work overlaps CPUInfer
  -> CANN host callback waits for CPUInfer completion
  -> asynchronous H2D of BF16 result
  -> later NPU operation verifies the result
```

This uses the Round 2A deterministic GGUF fixture through the production `KTMoEWrapper -> LlamafileMoEWrapper -> CPUInfer -> TP_MOE` path. It does not contain model integration or a hybrid expert implementation.

## Numerical result

- seed: `20260827` plus fixture offset
- input/output: BF16, shape `[1, 256]`
- experts/top-k: 4 / 2
- expert IDs: `[1, 3]`; routing weights `[0.7, 0.3]`
- max absolute error: `4.76837158203125e-07` (limit `1e-3`)
- mean absolute error: `1.862645149230957e-09` (limit `1e-4`)
- relative L2 error: `2.2967617095371596e-06` (limit `1e-2`)

## Stability

The full pipeline completed 1,000 cycles with all 1,000 terminal callback markers observed. In the final combined matrix RSS changed from `1,889,243,136` to `1,894,486,016` bytes: delta `5,242,880` bytes (5 MiB), below the 16 MiB acceptance threshold.

Result: `2 passed`; evidence: `runtime_pipeline_after_lifetime_fix.log`, `ascend_full_matrix.log`.
