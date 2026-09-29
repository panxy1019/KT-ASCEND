# B2 exact donor worker hook

Status: `PASS_OFFLINE_EXACT_HOOK_DEVICE_PENDING`; `live_eligible=false`.

The hook is revision-bound and default OFF. It does not edit the donor source.
An explicit worker bootstrap installs a reversible process-local wrapper on the
exact `vllm_ascend.worker.model_runner_v1.NPUModelRunner.load_model` class only
after the six-file B2 receipt and the broader 18-check source/config receipt
pass.

For the target `Qwen4ExpForConditionalGeneration`, the wrapper enters the
existing model-construction-scoped `qwen3_next.FusedMoE` installer before the
original `get_model` call. `_ascend_FusedMoE` therefore receives the matched
physical-Q32 `routed_experts_cls` and custom `runner_cls` before
`create_weights`. The original model loader and `AutoWeightsLoader` remain the
only checkpoint owners. After load, the hook resolves exactly 48
`loaded_model.language_model.model.layers[*].mlp.experts` instances, repeats
owner/runner/plan/mapper/shape/dtype checks, builds all lazy layer runtimes,
and commits the registry in one operation.

The runner overrides only upstream `_apply_quant_method`. Donor communication
prepare supplies the exact hidden/router objects used by the upstream selector.
The selector runs once. The device contract then orders KT submit, the native
preselected ALLGATHER weighted aggregate, KT sync and tensor-preserving merge.
The returned `[T,H]` partial contains original Top-K weights and no routed
scale. Upstream `MoERunner.forward` remains the sole scale/shared/output
transform/reduction owner, and the calling Qwen layer owns residual addition.
Ascend quant `apply` is never invoked by this path.

Run the read-only B2 source gate in the donor container after the existing
18-check preflight:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python3 run_b2_exact_worker_preflight.py \
  --vllm-ascend-root /opt/vllm/lib/python3.11/site-packages/vllm_ascend \
  --base-source-receipt /tmp/round16-b-p1-source-token.json \
  --output /tmp/round16-b2-worker-preflight.json
```

The expected status is `PASS_B2_EXACT_WORKER_SOURCE`, with six exact hashes,
`device_executed=false`, and `live_eligible=false`.

The first one-layer device-mode command has an actual installed factory name.
At the current milestone it deliberately stops before importing a model or
performing NPU work:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
ROUND16_B_ENABLE_P1=1 \
python3 -m round16_integration.p1_allgather_q32_probe \
  --enable-device \
  --source-token /tmp/round16-b-p1-source-token.json \
  --snapshot-manifest P1_REMOTE_SNAPSHOT_PREFLIGHT_MANIFEST.json \
  --binding-factory round16_integration.b2_unresolved_bindings:create_p1_runtime_binding \
  --tp-size 1 --ep-size 1 --enforce-eager \
  --dtype bfloat16 --quant-type NONE \
  --physical-experts 32 --logical-experts 512 --top-k 10 \
  --layer 0 --tokens 1 --max-model-len 256 \
  --output /tmp/round16-b-p1-device-receipt.json
```

Its expected result is the explicit
`SELECTIVE_Q32_PREALLOCATION_REQUIRED` exception. A successful device command
must use the same probe with a replacement factory that has the frozen
`round16-integration:p1-runtime-binding-factory:v1` identity and returns a
binding backed by the already-loaded B2 layer. No such successful factory is
claimed offline.

Remaining device gates are:

- Verify the physical-Q32 `RoutedExperts` constructor and Q32-only w13/w2
  checkpoint-slice loader before `create_weights`; the current source-bound
  scaffold deliberately lacks the live constructor identity.
- Run the six-source B2 and 18-check base preflights against the same live
  process/revision used for registration.
- Prove all 48 live owners retain physical32 BF16 w13/w2 and the full 512-entry
  map after `AutoWeightsLoader`.
- Prove current ALLGATHER comm and current stream identity, device tensor
  physicalization, KT submit/native/sync order, single routed scaling, and S15
  weighted aggregate parity.
- Prove template-inclusive context capacity at least 256 and the KT
  max-running-one cancel/drain/slot-reuse contract before REQUEST_ASYNC.

No NPU, model, checkpoint, C128 bank, graph capture, benchmark, or remote
container was used for this receipt.
