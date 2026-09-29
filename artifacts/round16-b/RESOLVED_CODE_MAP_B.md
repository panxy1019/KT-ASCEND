# Round16-B resolved code map

Status: `PASS_ROUND16_B_DEV_OFFLINE_CODE_MAP`

## Target identity

The only admitted architecture is `Qwen4ExpForConditionalGeneration`.  The
offline donor registers it as
`vllm_ascend.models.qwen4_exp:AscendQwen4ExpForConditionalGeneration` from
`vllm_ascend/models/__init__.py`.  The donor source identity supplied by the
contract is commit `550ab66e7feb89f6e2ba49394f6ffb153189ac2b`.

The hashed remote config independently selects
`Qwen4ExpForConditionalGeneration`; `Qwen4ExpForCausalLM`,
`Qwen3NextForCausalLM`, and every unrelated architecture fail closed. This
offline result is not device-qualified until the real-image source hashes and
the device Q32 RoutedExperts/checkpoint binding are verified.

## Local component map

| Contract responsibility | Local implementation | Resolved interface |
|---|---|---|
| Explicit model-only registration | `round16/b/model_plugin.py` | `ModelPluginRegistrar.register(registry, binding=VerifiedModelBinding(...))`; unbound calls return `VERSION_API_UNRESOLVED` without registry mutation |
| Frozen vLLM-Ascend 0.26 binding gate | `round16/b/native_vllm_adapter.py` | exact image/commit/source identity plus consumed observed-digest/API token, BF16/context launch gate, injected native paths/signatures, default OFF |
| Native registry class | `round16/b/native_vllm_adapter.py` | dynamic subclass of the injected Ascend model class; the native base constructor runs on the same object |
| Q32-before-native-construction seam | `round16/b/selective_preallocation_installer.py` | temporary target-Qwen `FusedMoE` wrapper injects verified class/immutable args; exact binding restored on every exit |
| Exact donor source preflight | `round16/b/native_source_preflight.py` | bounded SHA-256 then AST/config checks; default OFF; standard library only |
| Source-bound Q32 constructor/loader contract | `round16/b/source_bound_q32_adapter.py` | cloned 512-logical/32-physical state before allocation, w13/w2 binding records, rollback; deliberately live-ineligible |
| Immutable Top-10 split | `round16/b/routing_bridge.py` | `plan_routes(...) -> HybridRoutePlan` |
| Global/NPU-slot/CPU-logical ID separation | `round16/b/routing_bridge.py` | `RouteOccurrence.logical_expert_id`, `.npu_slot`, `.cpu_logical_expert_id` |
| Single weighting and merge | `round16/b/routing_bridge.py` | `merge_route_outputs(...)`; `routed_sum * scaling + shared_once` |
| Q32 allocation before reads | `round16/b/selective_loader.py` | `plan_q32(...) -> SelectiveLoadPlan` |
| Selective/fused/alias-safe load | `round16/b/selective_loader.py` | `SelectiveExpertLoader.load(...)` |
| CPU expert chunk and tail work | `round16/b/kt_expert_backend.py` | chunk/tail execution plus two-stage KT submit/sync; no native work ID permits one outstanding until successful drain |
| Full request/generation identity | `round16/b/lifecycle_bridge.py` | `RequestLease.state_lease: round16.common.contracts.StateLease` |
| Cancel/drain/late completion/slot reuse | `round16/b/lifecycle_bridge.py` | `RequestLeaseManager`; native allocation generation must strictly advance and invalid leases do not consume it |
| Graph/CPU replay boundary | `round16/b/graph_boundary.py` | `HybridGraphBoundary.replay(..., input_fingerprint=...)`; monotonic replay epoch and fingerprint enter CPU identity; failed replay/submit retires reserved tasks |
| Offline installation | `round16/b/install.py` | `python3 round16/b/install.py --target <empty-target>` |
| Exact donor worker hook | `round16/b/exact_donor_worker_hook.py` | exact-source token; constructor-scoped `NPUModelRunner.load_model`; atomic 48-layer runtime registration |
| Donor integration plugin | `round16_integration/b2_exact_worker_plugin.py` | paired owner/runner installer plus lazy current-context ALLGATHER tensor binding |
| Worker preflight | `round16_integration/b2_exact_worker_preflight.py` | read-only six-source SHA/API receipt layered on the 18-check base token |
| Executable unresolved gate | `round16_integration/b2_unresolved_bindings.py` | actual callable that raises `SELECTIVE_Q32_PREALLOCATION_REQUIRED` before model/device work |

## Offline donor and local evidence

The full selected-file path and SHA-256 map is in
`artifacts/round16-b/P0_B1_REAL_PATHS.md`; the machine-readable result is
`artifacts/round16-b/P0_B1_BINDING_RECEIPT.json`.

- `vllm_ascend/__init__.py` exposes the donor `register_model()` plugin path,
  but it also applies process-wide patches.  Round16-B does not reuse that
  global patch path.
- `vllm_ascend/models/__init__.py` proves the registry call shape
  `vllm.ModelRegistry.register_model(architecture, import_string)` and the
  exact target architecture/import string.
- `vllm_ascend/patch/worker/patch_fused_moe.py` reuses a platform-wide patch.
  Decision: `REJECTED_FOR_MODEL_SCOPED_PLUGIN`.
- `vllm_ascend/patch/worker/patch_process_weights_after_loading.py` replaces
  loader functions across loaded modules.  Decision:
  `REJECTED_FOR_MODEL_SCOPED_PLUGIN`.
- `artifacts/round14a/round14a_resident_runtime.py` freezes 512 logical experts,
  Q32 physical residency, compact physical mapping, and CPU logical IDs.
- `artifacts/round14a/FINAL_CONFIG.json` freezes the default placement as
  `STATIC_UNIFORM_Q32_BF16_NZ29` with dynamic/deferred migration disabled.
- Official vLLM `registry.py` at SHA-256 `27f6675a...da2f0` proves
  `ModelRegistry.register_model` accepts `type[nn.Module]`.
- Official `layer.py` at SHA-256 `c888133b...d725` proves `FusedMoE`
  accepts and invokes `routed_experts_cls` with `routed_experts_args` before
  weight creation.
- Official `qwen3_next.py` at SHA-256 `b642c10e...c74b` proves every sparse
  Qwen block calls its module-global `FusedMoE` during construction.
- Hashed model-config evidence resolves the served architecture as
  `Qwen4ExpForConditionalGeneration`, with 48 layers, 512 experts, Top-10,
  `bfloat16`, and `max_position_embeddings=262144`.

The B2 worker-source receipt additionally freezes donor `worker.py`
(`eb0bb11d...e0f8c0`), `model_runner_v1.py`
(`de9f108b...e2608`), Qwen4 wrapper `model.py`
(`b06a3123...81133`), the real Qwen4 consumer
(`a38101a1...9689`), `fused_moe.py` (`2d03cf69...1d47f7c`) and
`ascend_forward_context.py` (`af4a6e79...95c9f`). It resolves the safe window
as `NPUWorker.load_model -> NPUModelRunner.load_model -> get_model`: owner and
runner injection occurs before construction/allocation, while runtime factory
registration occurs only after `AutoWeightsLoader` completes.

The exact target instance path is
`loaded_model.language_model.model.layers.{layer}.mlp.experts` for
`Qwen4ExpForConditionalGeneration`. The former causal-only
`loaded_model.model.layers...` documentation is not used by B2 registration.

## Narrowed native boundaries

The hashed donor source resolves the registry mapping, the conditional model
constructor and `load_weights`, `NPUWorker`, `NPUModelRunner.load_model ->
get_model`, and `AscendUnquantizedFusedMoEMethod.apply`. The offline adapter
checks those injected paths and signatures without importing vLLM or torch.

The registry class-value path, Conditional architecture selection and
model-construction preallocation entry are now resolved from exact hashed
sources. The default-OFF installer wraps only
`vllm.model_executor.models.qwen3_next.FusedMoE` while the target native model
base constructor runs. It forwards to the already-resolved donor
`_ascend_FusedMoE`, injects a verified `routed_experts_cls` plus an immutable
Q32 mapping, checks all expected layers, and restores the original binding on
success or exception.

The source-bound Q32 adapter now proves with injected fixtures that a cloned
physical config is validated before native quant allocation, physical count is
32 while logical/global count remains 512, gate/up share `w13_weight`, down uses
`w2_weight`, nonresident IDs map to `-1`, and partial allocation is rolled back
while the primary error is preserved. This class is deliberately not accepted
by the live scoped installer.

The remaining runtime boundary is larger than RoutedExperts alone. The frozen
factory builds the original ExpertMapManager and `moe_config` first, and the
default AscendMoERunner consumes that original 512-local state after the
custom RoutedExperts instance. A physical-Q32 owner must therefore be paired
with a custom Hybrid runner in the same factory call. The registrar returns
`RUNTIME_RUNNER_BINDING_UNRESOLVED` without mutation for the current routed-only
installer; shared `moe_config` is not altered to conceal the mismatch.

The standalone preflight passed over the exact saved 17-source/config fixture.
It includes the actual Qwen consumer, upstream `MoERunner`,
`AutoWeightsLoader`, `default_weight_loader`, checkpoint remap, and donor
Ascend sources. A passing result issues a token containing observed digests and
API checks; native registration requires that exact token. It remains
`DEVICE_REQUIRED_NOT_RUN` inside the live image, and real checkpoint tensor
writes remain device pending.

The graph component defines only the ownership boundary: every replay validates
the lease before side effects, requires a non-empty input fingerprint, assigns
a monotonic replay epoch, and creates a fresh CPU identity containing the full
lease, epoch, and fingerprint.  NPU replay or CPU submission exceptions retire
the reserved task so drain and slot reuse remain possible.  No NPU graph was
captured or replayed here.  Device graph, real model, numerical, and performance
qualification remain `DEVICE_REQUIRED_NOT_RUN`/`NOT_RUN`.

## Latest fail-closed gates

The prepared Q32 manager is now validated for all 512 logical IDs before native
allocation. Model-construction acknowledgement repeats that validation against
the live manager and requires distinct physical32 BF16 `w13_weight` and
`w2_weight`. The launch contract requires 128 input plus 128 output capacity,
with `max_model_len` and state capacity each at least 256.

Because the saved KT submit/sync ABI has no native work ID, the backend permits
one outstanding FIFO item. A second submit or stale/reversed handle is rejected;
a sync failure retains the item until a successful real drain. Device proof of
this sequence is still required before REQUEST_ASYNC. Actual Qwen
`self.experts` runner wiring, native preselected-Top-K execution, and S15
weighted aggregate parity remain the minimum REAL_MOE blockers.

## Qwen `self.experts` source-bound runner milestone

`build_source_bound_native_runner_class()` now produces the construction-time
runner class paired with the Q32 routed owner. The scoped installer supplies
both classes and immutable args in one `qwen3_next.FusedMoE` call, validates
that the runner retains the exact owner and plan, and repeats those checks for
all 48 layers at acknowledgement.

The generated runner wraps upstream `_apply_quant_method` with an injected
`MoECommMethod.prepare` result. The prepared hidden states and router logits
flow into the upstream selector and routed owner. Selection occurs once; the
Ascend quant `apply` path is never called. Prepared invalid rows are converted
to safe expert ID 0 and zero weights before CPU or native payloads. The current
comm object must retain identity from prepare through fused execution.

The source-bound native result is a token-level `[T,H]` aggregate with original
Top-K weights already applied and routed scaling still absent. Per-occurrence
native mappings belong only to the legacy CPU arithmetic mock and are rejected
by the Qwen runner path. The inherited `MoERunner.forward` owns routed scaling,
shared handling, output transform, reduction and zero-expert output; the Qwen
caller owns residual addition.

The exact-source gate now covers 17 source files plus model config. New donor
members are `moe_comm_method.py` (`043f3c...`), `prepare_finalize.py`
(`c467d0...`), `quant_type.py` (`2faa44...`) and `token_dispatcher.py`
(`36fa36...`), `moe_runtime_args.py` (`c1d523...`) and
`ascend_forward_context.py` (`af4a6e...`). The first device candidate is limited to TP1/EP1 eager
ALLGATHER, BF16/QuantType.NONE, physical32, `expert_map=None`, and
`log2phy=None`. Live execution remains ineligible until the device adapter,
native owner/runner pair, live source token, resident receipt, and numerical
parity gates close.

## P1 ALLGATHER physical-Q32 tensor contract

`allgather_q32_device_adapter.py` separates the offline arithmetic shim from
the device tensor contract. `AllGatherQ32OfflineFakeAdapter` is the only path
that converts rows into Python tuples and floats. It covers all-CPU, all-NPU,
mixed and padded route masking without claiming device behavior.

`AllGatherQ32DeviceTensorAdapter` receives opaque tensor objects and reads only
shape, dtype and device metadata. It requires prepared BF16 NPU hidden states,
int32 NPU Top-K IDs, BF16 NPU Top-K weights, exact current comm/stream identity,
and an identity-bound physical payload callback. The callback contract maps
resident logical IDs to slots `0..31`, changes CPU/padding occurrences to
zero weight and safe ID 0, and returns tensors on the same NPU. The adapter then
calls the exact builder, `fused_experts`, and `finalize` and returns the same
finalized weighted/unscaled `[T,H]` device tensor without iteration, scalar
extraction, host conversion, or synchronization.

The device result has a distinct API identity from the existing tuple-based
runner result. This prevents accidental use of a Python-row merge in the live
path.

## Executable integration probe milestone

The documented integration entrypoint now exists at
`round16_integration.p1_allgather_q32_probe` and is included by the standard
library installer. Importing it has no torch, torch_npu, vLLM or device side
effect. Its `--validate-only` mode consumes the exact 18-check source receipt,
the bound S0 v295 snapshot manifest and the packaged 130 KiB
`q32-physical.json` metadata receipt before any runtime factory import.

`DevicePhysicalTopKPayloadBuilder` implements the live tensor algorithm through
an injected torch-compatible operation set: safe-clamp logical IDs, gather the
int32 512-entry map, form resident and valid-row masks, select physical IDs,
and zero CPU/invalid/padding weights. `TensorWeightedAggregateMerger` adds the
native and KT BF16 NPU `[T,H]` weighted/unscaled tensors. Opaque tensor spies
reject iteration, `item`, `tolist` and truth extraction; the tested code uses
none of them and calls no synchronization API.

Before the first probe side effect, `validate_live_owner` checks the requested
layer path, identical plan, exact 512-entry manager mapping, distinct NPU BF16
weights with S0 shapes `[32,2560,1280]` and `[32,640,2560]`, plus current comm
and stream identity. The executable order is prepare, selector once, KT submit,
native physical-Q32 fused envelope, KT sync and tensor add. Failure before
native execution retires prepared state; native or sync failure invokes the
identity-bound KT abort/drain callback while preserving the primary exception.

The remaining offline-unresolved seam is the donor worker runtime factory. The
frozen source does not expose a safe process-global loaded-model and stream
handle for a standalone process. Device mode therefore requires an explicit
identity-bound factory supplying the already-loaded layer, current comm/stream,
exact builder/selector and KT callbacks. Omitting it fails before any runtime
import or native execution; no generic factory is fabricated.
