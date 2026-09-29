# Round16-B runtime seam review

Status: `SOURCE_BOUND_REVIEW_COMPLETE_HYBRID_RUNNER_STILL_UNRESOLVED`.

This review uses the frozen vLLM commit
`568afb3a13806beb53bb2e6bd518269357b237c0` and vLLM-Ascend commit
`550ab66e7feb89f6e2ba49394f6ffb153189ac2b`. It does not claim a device run.

## Construction order that matters

The upstream `FusedMoE` factory constructs objects in this order:

1. one `ExpertMapManager` from the model's global expert count and parallel
   configuration;
2. `RoutedExperts`, which calls `create_weights` in its constructor;
3. the selected `runner_cls`, receiving the same `moe_config`, router,
   `RoutedExperts`, shared-expert modules and transforms.

For TP1/EP1 with the unmodified Qwen configuration, the upstream manager and
`moe_config.num_local_experts` describe all 512 routed experts. The default
`RoutedExperts` therefore allocates 512 `w13_weight` and `w2_weight` rows.

The model-construction-scoped installer is a valid early interception point,
but injecting only a custom `routed_experts_cls` cannot by itself complete the
Hybrid port. The frozen Ascend runner validates its local expert capacity
against its placement result, allocates runtime tensors from
`moe_config.num_local_experts`, and passes that same count into communication
and token-dispatch metadata. A class that physically allocates 32 rows while
silently leaving the default runner on a 512-local-expert contract would have
inconsistent storage and runtime metadata. Mutating the shared `moe_config`
inside the weight owner would also change the runner's logical model contract
and is rejected.

## Required bounded seam

The real B implementation therefore needs a matched pair, injected through
the already-supported `routed_experts_cls` and `runner_cls`/`runner_args`
arguments:

- a source-bound Q32 weight owner that keeps logical expert count 512, owns
  exactly 32 physical BF16 resident rows, exposes the frozen global-to-local
  map, and makes the checkpoint loader skip non-resident rows before any NPU
  destination allocation;
- a Hybrid runner that invokes the existing router exactly once, freezes the
  original Top-10 IDs and weights, sends resident occurrences to the native
  Q32 path and non-resident occurrences to the existing C128/MXFP4 backend,
  then performs one complete same-layer merge before the next layer consumes
  the result;
- one explicit location for routed scaling, shared-expert computation, shared
  gate and residual application. Each must execute exactly once;
- one work ID per non-empty CPU branch and request/layer/step generation checks
  so cancellation drains in-flight work and discards late results;
- eager execution first. CPU submit/wait/merge remains outside any later NPU
  graph segment.

The upstream `RoutedExperts.weight_loader` already maps global checkpoint
expert IDs through its expert map and can skip IDs mapped to `-1`, but the
custom physical-32 owner must prove this behavior for the actual fused
`gate_up_proj`/`down_proj` names, TP1 shard rules, `w13_weight`/`w2_weight`
shapes, aliases and `AutoWeightsLoader` call sequence. It may not call the
default constructor first and shrink the allocation afterward.

## Current boundary

The scoped registry/model-construction hook is offline qualified and remains
default OFF. The Q32 checkpoint owner, matched Hybrid runner, native resident
execution, CPU merge, cancellation and full-layer numerical comparison are
`DEVICE_REQUIRED_NOT_RUN`. Until both injected classes are verified together,
the B path is an integration candidate rather than a runnable vLLM-KT model.
