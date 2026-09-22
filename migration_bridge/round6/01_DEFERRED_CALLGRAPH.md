# Deferred Expert Call Graph

Evidence state: `CODE_INSPECTED`

## Construction

```text
ServerArgs.kt_max_deferred_experts_per_token
  -> create_kt_config()
  -> KTConfig.max_deferred_experts_per_token
  -> KTEPWrapperMethod.create_weights()
  -> layer_max_deferred
  -> KTMoEWrapper(... max_deferred_experts_per_token=...)
  -> BaseMoEWrapper.max_deferred_experts_per_token
```

Current child behavior only forces `layer_max_deferred=0` for the final model
layer.  It does not inspect whether the immediately following layer has a KT
wrapper.

## Submit

```text
KTEPWrapperMethod.apply()
  -> staging_buffer.copy_(x)
  -> _submit_with_staged_input()
  -> _submit_cpu_forward()
  -> BaseMoEWrapper.submit_forward()
  -> _prepare_forward_cpu_buffers()
       current_slot = layer_idx % 2
       next_slot = (current_slot + 1) % 2
       select_deferred_experts()
       framework copy_ to pinned CPU buffers
  -> NPU: _wait_device()
  -> immediate forward_task(incremental=previous_layer_pending)
  -> cpu_infer.submit(immediate)
  -> deferred forward_task(output=next_slot, incremental=False)
  -> cpu_infer.submit(deferred)
  -> layer_pending=True
```

On NPU both immediate and deferred work are submitted synchronously from the
host into the CPUInfer queue after the framework D2H is made visible.  The
WorkerPool still overlaps with independent NPU expert work.

## Synchronize and merge

```text
KTEPWrapperMethod.apply()
  -> NPU expert apply()
  -> _sync_with_staged_input()
  -> BaseMoEWrapper.sync_forward()
       allow_pending = 1 when current layer submitted deferred work
       _wait_device(NPU)
       cpu_infer.sync(allow_pending)
       output_gpu[current_slot].copy_(output_cpu[current_slot])
  -> FP32 NPU partial + FP32 CPU partial
  -> final model BF16-visible boundary
```

At layer `L+1`, `incremental=True` makes the CPU kernel add the contents of its
caller-provided output buffer before adding the new immediate CPU result.  With
the two-slot mapping, `L` deferred output and `L+1` current output target the
same slot.

## Selection semantics

`select_deferred_experts()` runs `torch.topk` over routing scores, but builds a
single expert-ID protection flag across the full flattened batch.  Therefore
prefill behavior is batch-union protection, not a strict guarantee of exactly
one deferred CPU expert per row.  The CPU kernel subsequently filters work by
the CPU/GPU ownership mask.

This must be captured as explicit telemetry; CLI value `1` is only a maximum
selection budget, not an actual per-token deferred count.

## Historical source

- Deferred core: upstream `9bc00e587ba57442b1e3398ce0abd5a5675ac55a`.
- LLAMAFILE deferred race fix: `e7d1c1de09b375178eaf94db8749e36f8de3da2c`.
- Round 5 NPU transfer/callback semantics remain the immutable local baseline.

## Gate

```text
R6_G1_CALLGRAPH = CODE_INSPECTED
```
