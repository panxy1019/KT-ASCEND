# Dynamic BF16 full-context compatibility repair

2026-09-17. Original private runtime and failed attempts remain immutable.

## A3 reproduction

Corrected direct R64 exactly reproduced174/256, zero invalid; its resolved argv
matches the historical direct Ascend reference. The driver then failed during
the first threshold256 triggered request, before migration publication.
`SharedFullContext._prepare_weight_bf16` called
`os.sched_setaffinity(0, {255})`, but this container permits CPU128–191 only.
The scheduler raised OSError22 and its owned guard reaped all model children.
Initial short/no-trigger request completed. No later threshold or HD ran.

## Official source audit — CODE_INSPECTED

The pinned official [SGLang BF16 transport](https://github.com/kvcache-ai/sglang/blob/3424f35d0e60f03b9e75d9b90941f8f4cb944524/python/sglang/srt/layers/moe/kt_ep_wrapper.py)
also derives a target core from host CPU count and hardcodes CUDA stream APIs.
This raw branch is not container-cpuset/Ascend portable as written. This does
not justify an upstream wholesale upgrade or threshold suppression.
[PyTorch2.9 get_device_module](https://docs.pytorch.org/docs/2.9/generated/torch.get_device_module.html)
provides dispatch to the tensor's actual device module.

## Minimal scoped port

Only `_prepare_weight_bf16` changes: preserve caller CPU affinity instead of
pinning to a host-global last core; select stream/event API using the destination
tensor's device type. Keep official double-buffer writes, event fences,
Tensor.copy_, expert ownership, full-context prefill and hot-set update unchanged.
No casts, weights, masks, thresholds or model arithmetic change. INT4/FP8 paths
remain untouched and out of scope. No new Hybrid coordinator is introduced.

`bf16_context_compat.py` AST-locates the single method and rejects source drift.
`prepare_bf16_context_runtime.py` checks the actual imported wrapper SHA,
copies the SGLang Python source into a private owned runtime, patches only that
method, and creates fresh provenance/config/profile inputs without copying PASS
results. Original deployed source and native extension are not overwritten.
Launch import precedence must explicitly match the frozen private runtime/hash.

Regression uses a fake NPU API and forbidden affinity mutation, verifies event
waits, unchanged unrelated CUDA code, source-drift rejection and provenance
binding. Local tests are not A3 system qualification. Next re-run predecessor
compact/reference gates and the original triggered probe in a new directory.

## Preserved evidence and new execution

Lossless whole v2 corrected-R64/trigger failure/process/provenance/tool archive:
`raw/direct-r64-and-dynamic-affinity-failure-v1.tar.gz`, SHA256
`a8521b8ad6738add53625e5263127b8cc93036d03f887c0a68638add3437b244`.
Complete SCPexit0; local/A3 hashes equal; full tar validationexit0.
Private runtime-v4 wrapper SHA
`ea7fe7a6c46edd6eb0f6225a90f0892e0375a7752a9e7720c36711cf099cfcc5`.
Original source stays8eec41...; repaired CPU native stays461821... .
Tools-v15 passed89 pure-Python tests locally and on A3. The new driver
`backing-requalification-v3` reruns operators, both Static compact lanes and
direct R64 before the original three DEV probes. Launch is not gate completion.
