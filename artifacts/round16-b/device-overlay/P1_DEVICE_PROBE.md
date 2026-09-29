# P1 TP1/EP1 ALLGATHER physical-Q32 probe

The executable entrypoint now exists at
`round16_integration.p1_allgather_q32_probe`. Importing it does not import
`torch`, `torch_npu`, vLLM, or vLLM-Ascend. Device execution is default OFF.

The static gate is directly runnable in the donor container after installing
or copying the Round16 package and integration overlay. First produce the exact
source receipt:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python3 round16_b_native_source_preflight.py \
  --enable \
  --vllm-package-root /opt/vllm/lib/python3.11/site-packages/vllm \
  --vllm-ascend-package-root /opt/vllm/lib/python3.11/site-packages/vllm_ascend \
  --model-config /model/config.json \
  --output /tmp/round16-b-p1-source-token.json
```

Then validate the source token, S0 Q32 evidence, layer-0 shape/map receipt and
runtime requirements without importing a device runtime:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python3 -m round16_integration.p1_allgather_q32_probe \
  --validate-only \
  --source-token /tmp/round16-b-p1-source-token.json \
  --snapshot-manifest artifacts/round16-b/device-overlay/P1_REMOTE_SNAPSHOT_PREFLIGHT_MANIFEST.json \
  --output /tmp/round16-b-p1-static-receipt.json
```

The expected status is `READY_FOR_DEVICE_RUNTIME_BINDING`, with
`device_executed=false`, `source_checks=18`, and the loaded-owner path
`loaded_model.language_model.model.layers.0.mlp.experts.routed_experts`.

The bounded device invocation uses the same existing module. The shipped
factory name below is real and deliberately fails before model/device work at
the still-open physical-Q32 preallocation gate:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
ROUND16_B_ENABLE_P1=1 \
python3 -m round16_integration.p1_allgather_q32_probe \
  --enable-device \
  --source-token /tmp/round16-b-p1-source-token.json \
  --snapshot-manifest artifacts/round16-b/device-overlay/P1_REMOTE_SNAPSHOT_PREFLIGHT_MANIFEST.json \
  --binding-factory round16_integration.b2_unresolved_bindings:create_p1_runtime_binding \
  --tp-size 1 --ep-size 1 --enforce-eager \
  --dtype bfloat16 --quant-type NONE \
  --physical-experts 32 --logical-experts 512 --top-k 10 \
  --layer 0 --tokens 1 --max-model-len 256 \
  --output /tmp/round16-b-p1-device-receipt.json
```

The expected current result is
`SELECTIVE_Q32_PREALLOCATION_REQUIRED`, before a model import or NPU call. A
successful replacement binding callable must have API identity
`round16-integration:p1-runtime-binding-factory:v1` and return
`P1RuntimeBinding`. It supplies the already-loaded layer owner, exact current
`AllGatherCommImpl`, current NPU stream, exact fused-input builder, prepared
hidden/router tensors, one selector callback, and KT submit/sync/abort-drain
callbacks. No generic built-in donor factory is claimed: the saved exact source
does not expose a process-global loaded-model/stream handle that a standalone
CLI can safely discover. Missing factory, identity, live owner, comm, stream,
manager map, or source token fails before native execution.

The one-layer weight fixture is not reread from a checkpoint or expert bank.
The factory receives the already-loaded donor owner at
`loaded_model.language_model.model.layers.{layer}.mlp.experts.routed_experts`. Before any
probe side effect, the executable checks the S0 v295 receipt identity, exact
512-entry map, distinct NPU BF16 `w13_weight`/`w2_weight`, and exact shapes
`[32,2560,1280]` / `[32,640,2560]`. The S0 receipt proves layout and placement,
not tensor bytes or numerical parity.

The implemented physical payload callback uses tensor operations only:
logical IDs are bounded for safe lookup, gathered through the 512-entry device
map, CPU/invalid/padding routes receive zero weight and physical placeholder 0,
and resident routes retain their Top-K weights. The callback calls no
`item`, `tolist`, tensor iteration, host copy, or synchronization API. Native
and KT weighted `[T,H]` partials are added as tensors, with routed scaling left
to upstream `MoERunner.forward`.
