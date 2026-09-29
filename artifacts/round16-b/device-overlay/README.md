# Round16-B read-only donor preflight overlay

`round16_b_native_source_preflight.py` is a byte-identical, standard-library
copy of `round16/b/native_source_preflight.py`. It is default OFF, performs
bounded reads, compares exact SHA-256 values before AST checks, and never
imports or mutates vLLM, torch, torch_npu, the model, or a device.

```text
python3 round16_b_native_source_preflight.py \
  --enable \
  --vllm-package-root <site-packages>/vllm \
  --vllm-ascend-package-root <site-packages>/vllm_ascend \
  --model-config <model-revision>/config.json \
  --output <temporary-output>/round16-b-preflight.json
```

Exit code 0 and `PASS_EXACT_SOURCE_AND_API` bind exactly 17 reviewed sources
plus the model config. The gates include registry class acceptance, the Qwen
model-construction seam, owner+runner forwarding, upstream MoERunner routing
and post-forward ownership, RoutedExperts/create_weights, the real
AutoWeightsLoader consumer, checkpoint name remap, Ascend internal routing and
Ascend local-expert runtime metadata, the exact `MoEFusedExpertsInput` builder,
and the forward-context ALLGATHER comm identity. The config gate also requires
`dtype=bfloat16` and `max_position_embeddings>=256`. A passing preflight issues
an observed-digest/API token; live registration must consume that token.

This source result does not enable a live Hybrid. The CPU-only source-bound
runner and atomic paired installer remain `live_eligible=false` because the
current Ascend quant `apply` routes internally and the default Ascend runner
constructs placement/EPLB/runtime state from the original local512 contract.
The overlay includes a default-OFF EP1/TP1 ALLGATHER tensor contract and the
actual `round16_integration.p1_allgather_q32_probe` entrypoint. Its physical
Top-K builder and KT/native merger use injected tensor operations and do not
materialize tensor values on the host. `--validate-only` binds the exact source
receipt and S0 snapshot without importing a device runtime. Device mode still
requires a reviewed donor worker factory that supplies the already-loaded
layer, current comm/stream and KT callbacks; the saved source has no safe
process-global loaded-model handle. The offline tuple fake is not a live tensor
implementation. No shared `moe_config` mutation is an acceptable substitute.

The overlay does not prove the launch-time context allocation. The subsequent
TP1 eager gate must show template-inclusive input <=128, output <=128, and both
`max_model_len` and state capacity >=256. The KT native ABI also has no work ID;
REQUEST_ASYNC remains blocked until the real stream proves max-running-one
submit/native/sync, cancellation drain, and slot reuse.

## B2 exact worker hook

`run_b2_exact_worker_preflight.py` validates six additional pristine donor
sources that freeze the worker load, Qwen4 consumer, MoE runner and
forward-context entrypoints. `round16_integration.b2_worker_overlay_entry` is
the default-OFF bootstrap, while `b2_exact_worker_plugin` implements
constructor-time paired owner/runner injection and post-load atomic 48-layer
registration. Current comm and stream are resolved only inside an active
forward.

The installed `b2_unresolved_bindings` module provides real callable factory
names and fails at `SELECTIVE_Q32_PREALLOCATION_REQUIRED`. This is the precise
offline stop: the donor image still needs a verified physical-Q32
`RoutedExperts` constructor/checkpoint-slice loader. No successful live
physical32 load or device execution is claimed.
