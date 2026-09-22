# Expert shadow prepare provider and fixture contract

Status: **E0 CONTRACT COMPLETE / PHYSICAL EXECUTION NOT CLAIMED**. This document is a frozen-source and existing-evidence audit. It does not report a new component run, a captured Prefill row, or a behavior optimization.

## 1. Source states and immutable identities

The provider files audited here are under `migration_bridge/round9/`; the short `runtime/...` and `tests/...` names below are their paths relative to that directory.

| State | Identity | Meaning |
|---|---|---|
| v61/baseline | commit `57de93dfbb5775f1265378c1eae737edb8d5cc3e`; source-tree SHA256 `8f68fb1157353c0e95c2e52ad93c8a2136d1d0a7dce8867324a3d17f0d6f2099` | Frozen execution baseline in `baseline-lock.json`. |
| v63 observation build | source-tree SHA256 `12cf0c95651ea8b5b75d599b1d440cf62e74d3fa6b6df572d775003ce2aa2094` | Build recorded by `evidence/new-build-receipt.json`; produced the normal Prefill timeline and decode capture. |
| corrected replay snapshot | `evidence/replay-source-sha256.json` | Separate v20 component snapshot. It is not the v63 build. Its corrected `tests/q32_ktep_probe.py` SHA256 is `d64eb6bbee571376ea86b1e9a4a37fbdd729166c80897e0b3f509ecf3fda18db`; baseline was `e395b855c9ed471c991df5d88b75502fef5dd4bd719fceed5a4042f4a790d80b`, and the v63 receipt recorded the intermediate observation-build copy as `91ccd88b96c7b5abacd98826b354043f870f62af5110560690b374386504cfcf`. |

At the time of the E0 audit, the five provider/scaffold files requested below were byte-identical between the baseline, v63 receipt, and corrected replay manifest. These hashes describe the immutable historical provider/scaffold inputs. They do **not** describe the subsequently edited current checkout: the revised decomposition adds default-OFF observation seams in `runtime/dynamic_adapter.py` and `runtime/expert_source.py`, plus the new `runtime/shadow_decomposition.py`. Those edits require their own final snapshot hashes and must not be reported as baseline-identical.

| File | SHA256 |
|---|---|
| `runtime/prefill_binding.py` | `8188e20c4e1a5ebb53a22caa6f76c46e2fb09a90376b7f8bf91933676e160df7` |
| `runtime/layerwise.py` | `6647b1ef56b80941b49222552a13746d519140613026f3bb6a3d1b83cceccbc6` |
| `runtime/dynamic_adapter.py` | `6fb297ea20abfeb945d56d63b72e07731032ca93802c16b212e46dc3cb1e440a` |
| `runtime/expert_source.py` | `98f51611f4f8ad4c926b6095a601a7d894d3cd6344e0e3b2565bb6c6d2d654bd` |
| `tests/layerwise_device_probe.py` | `0bb9fd863cf0f0ef0626fb253432d72ac19c9bbac3ae4947685d13009c098af8` |

The canonical model/source identity is revision `de4b8e4d43b917e7706784d8bb445c9af86a3540`, model manifest SHA256 `62e6eca0f2900745dbfd9ba6ed76e60547e0d602acf2d62a9ecac3972028b38e`, verification receipt SHA256 `7d6cb3db19497303bdd8ceb24efae626e1926a610ab7fbd4406b47d05db4a239`, and model path `/models/de4b8e4d43b917e7706784d8bb445c9af86a3540`. The layer-0 MXFP4 CPU-bank SHA256 is `e9d6b27f352d501a9ed3737ec5d3ab140b9d1f5bda22d2e36ef9ef0bcc45b08b`; it is a separate source from the canonical BF16 checkpoint.

## 2. Exact Prefill provider call chain

The reported v63 `shadow_prepare` totals, 60.2026 seconds over 65 calls in the first sampled Prefill and 264.2664 seconds over 274 calls in the normal 64-token Prefill, correspond to the Host boundary in `prefill_binding.bind_expert_timeline`, specifically its local `AdapterObserver.copy_experts`. It wraps the already-created `backend.adapter` and times one call to the real copy facade. The event's accounted bytes are `copied_bytes_after - copied_bytes_before`; they are logical BF16 expert payload bytes, not measured link traffic.

The actual call chain is:

```text
scheduler EXTEND hook
  prefill_binding.prefill_before
  -> PrefillRuntime.before
       requires serial request, EXTEND mode, total >= configured threshold,
       chunk count 1..32, no prefix sharing, and establishes request lease
  -> KTEPWrapperMethod.apply replacement installed by bind_expert_method
  -> PrefillRuntime.apply
  -> LayerwiseTransport.execute_layer
  -> group_topk_routes
  -> AscendBF16TransportBackend.execute_group
  -> AdapterObserver.copy_experts                 [v63 shadow_prepare parent]
  -> _CopyOnlyAdapter.copy_experts
  -> TorchKTEPAdapter.copy_experts                [real prepare provider]
  -> AscendBF16TransportBackend.method.apply      [shadow compute]
  -> LayerwiseTransport.execute_layer:
       backend.synchronize(identity)
       backend.release(identity)
```

`PrefillRuntime.before` enters this path only for an admitted plain EXTEND batch. Decode returns without activating the runtime; `bind_expert_method` then calls the original mixed KTEP `apply`. Therefore a resident decode invocation is not the observed Prefill provider. The captured row described below is legal only as component input to the real BF16 shadow provider.

`prefill_model_ready` also proves the production construction chain: it validates the device gate and 48 real KTEP layers, constructs `TransportReservation`, creates `AscendBF16TransportBackend`, and passes it to `LayerwiseTransport`; the transport constructor calls `backend.preallocate`. The timeline adapter is bound only after preallocation has created the private shadow and copy facade.

## 3. Canonical source, materialized tensors, and layout

`AscendBF16TransportBackend.preallocate` constructs `OriginalExpertSource` against the immutable model path. `OriginalExpertSource.read(layer, expert, projection)` resolves `model.safetensors.index.json`, checks the shard remains under the model root, caches and revalidates the safetensors header with `(size, mtime_ns, inode)`, requires BF16 and the exact checkpoint shape, then opens/seeks/reads exactly one projection slice.

For hidden size 2560 and intermediate size 640, one projection slice is 3,276,800 bytes. One logical expert is three slices, 9,830,400 bytes. This is original checkpoint BF16; no MXFP4 read or dequantization participates in shadow prepare.

For an `n`-expert group, `TorchKTEPAdapter.copy_experts` does the following for each persistent destination projection:

1. Allocate an NPU **ND format 2** staging tensor. `w13` staging is `[n, 2560, 1280]` BF16 (`6,553,600 * n` bytes); `w2` staging is `[n, 640, 2560]` BF16 (`3,276,800 * n` bytes).
2. Materialize CPU BF16 tensors with `torch.frombuffer(bytearray(raw))`. For `w13`, read gate and up separately, concatenate them into a new `[1280, 2560]` CPU tensor, and use its transposed view as the copy source. For `w2`, transpose the `[2560, 640]` CPU tensor as a view. The wrapper/view is not itself an NZ conversion.
3. Copy each source into the ND staging row with `non_blocking=False`.
4. Call `destination.index_copy_(0, index, staged)` on the persistent **NZ29** shadow parent. This is the ND-to-NZ publication operation; the destination remains the same persistent parent storage, while `staged` is a new temporary allocation. The existing code does not expose a separate conversion object or event.
5. Call the adapter's existing device-wide `synchronize()` after each projection, check that the persistent destination still has format 29, then delete the staging reference.

The persistent shadow parents are distinct from the resident template and have these exact shapes and storage bytes:

| Parent | Shape | Dtype/layout | Bytes |
|---|---:|---|---:|
| `shadow.w13_weight` | `[32, 2560, 1280]` | BF16 / NZ29 | 209,715,200 |
| `shadow.w2_weight` | `[32, 640, 2560]` | BF16 / NZ29 | 104,857,600 |
| Private 32-slot bank | | | **314,572,800** |

The maximum single ND projection staging allocation is 209,715,200 bytes. The **historical broad layerwise scaffold** admitted a 1 GiB workspace budget, so its old component reservation was `314,572,800 + 209,715,200 + 1,073,741,824 = 1,598,029,824` bytes. The gate measured constructor peak extra 524,289,536 bytes and execution peak extra 616,490,496 bytes. Production `prefill_model_ready` instead computes its reservation with the larger of those historical measured peaks, yielding `314,572,800 + 209,715,200 + 616,490,496 = 1,140,778,496` bytes.

The **current revised A/B component configuration** uses a deliberately narrower 512 MiB workspace allowance: `314,572,800 + 209,715,200 + 536,870,912 = 1,061,158,912` reserved bytes. This is the configuration to record for the revised probe. The old 1 GiB scaffold reservation, the revised 512 MiB component reservation, and the production full-model reservation are three distinct contexts of the same formula and must not be silently interchanged. The revised run must still reject any observed allocation above its 512 MiB workspace allowance.

## 4. Existing completion, lease, and retire semantics

The observed v63 `shadow_prepare` parent already includes both projection-level device-wide synchronization waits because `TorchKTEPAdapter.copy_experts` calls `self.synchronize()` after each `index_copy_`. The provider therefore returns only after both staged projection updates are complete under the current implementation. A new observer must identify enqueue and completion separately without adding a new wait.

After prepare, `AscendBF16TransportBackend.execute_group` constructs row selection and remapped IDs, calls the private unquantized fused-MoE method, and returns result-row views. `LayerwiseTransport.execute_layer` then calls `backend.synchronize(identity)`, which device-synchronizes the compute and marks `_synchronized = identity`. Only after that does `backend.release(identity)` clear `_current` and `_synchronized`. The next group is serial and cannot acquire the shadow bank while `_current` is live.

There is no separate source-reader retirement queue or slot event. Each checkpoint file context closes at the end of `OriginalExpertSource.read`; local `raw`, CPU tensors, concatenation result, and ND staging become releasable only after their enclosing copy step and its existing synchronization. Old slot contents remain in the persistent NZ29 parent after release and are overwritten by the next full prepare. They are not a published cache and carry no reusable key. On error, transport synchronization/release failure poisons the backend; abort device-synchronizes any live `_current` lease before clearing it.

The current identity is `BufferIdentity(request_id, placement_epoch, allocation_epoch)`. A future shadow reuse key would additionally have to bind at least model revision/receipt, logical layer, canonical BF16 source/index identity, source layout, destination NZ29 layout and geometry, ordered logical expert-to-slot mapping, and slot generation. This contract does not add reuse.

## 5. Numerical branches and references

The captured fixture manifest is unambiguously a **decode** row: layer 0, `mode=decode`, one token, hidden `[1,2560]` BF16, top-k IDs `[1,10]`, top-k weights `[1,10]` FP32, actual resident IDs 0–31, and NZ29 resident weights. Its tensor bundle SHA256 is `de82a710a2fd56c7587a91dbc5c37b8989ff0265a502391d797588c77c6e8fc6`.

Its recorded output and v20 exact replay belong to the original mixed branch: resident experts 20 and 14 used original BF16 NPU weights; the other eight assignments used the MXFP4 CPU bank with Q8 activation handling; FP32 partials were merged and cast once to BF16. The output SHA256 `0851f485287fca4f258ac946482d32d807e3220dfa5ace0398860198f5dd7897` is valid for that mixed branch only.

The shadow branch loads canonical BF16 for all routed experts and runs the private BF16 NZ29 NPU method. Its reference is the all-BF16 CPU oracle already implemented by `tests/layerwise_device_probe.py:cpu_reference`: original BF16 gate/up/down, BF16-rounded gate/up projections, BF16-rounded activation, BF16-rounded down projection, FP32 weighted accumulation, and final BF16 cast. The existing qualified tolerance is relative L2 <= 0.01, with finite BF16 output and exact post-copy shadow-weight identity. Neither shadow case below may be compared bit-for-bit to the mixed captured output.

## 6. Frozen A and B component cases

### A — captured decode-input-driven BF16 shadow

Use the captured `hidden_states`, `topk_ids`, and `topk_weights` without modification:

```text
ordered routes: [20, 150, 14, 111, 203, 414, 211, 358, 33, 230]
weights: [0.3745466471, 0.1510056108, 0.0962743163, 0.0761622190,
          0.0715463236, 0.0540718772, 0.0474777445, 0.0474777445,
          0.0407187603, 0.0407187603]
sorted shadow slots: [14, 20, 33, 111, 150, 203, 211, 230, 358, 414]
```

Label this `CAPTURED_DECODE_INPUT_DRIVEN_SHADOW`, not captured Prefill. The real `group_topk_routes` must construct the group, and the revised component probe must call the original `AscendBF16TransportBackend.execute_group`, followed by that backend's original `synchronize(identity)` and `release(identity)`. This direct backend lifecycle proves the component provider/consume/retire sequence, but it does not exercise the scheduler, `PrefillRuntime`, a whole 48-layer `LayerwiseTransport` request, or full Prefill. Direct adapter-only copying remains insufficient because it would omit compute and lease retirement. Its numerical oracle is a newly evaluated all-BF16 A reference.

### B — one missing-expert replacement

Create one explicitly derived component case by changing only the last ordered route from expert 230 to expert 231. Keep the captured hidden state and all ten captured route weights unchanged:

```text
ordered routes: [20, 150, 14, 111, 203, 414, 211, 358, 33, 231]
sorted shadow slots: [14, 20, 33, 111, 150, 203, 211, 231, 358, 414]
case label: DERIVED_COMPONENT_CASE
```

Expert 231 is in range, absent from A, and resolves through the same layer-0 canonical BF16 source and NZ29 destination. Sorting changes only slot 7 (`230 -> 231`), so A-to-B has exactly nine retained logical slot identities and one outgoing/one incoming expert. This is the narrowest legal missing-expert transition. The current provider must still perform the full ten-expert prepare; the case must not be described as incremental copy or reuse. B requires its own all-BF16 CPU reference and output hash. A's mixed captured output and A's all-BF16 reference are both invalid as B's oracle.

For each case, freeze separate IDs for model/source, layer, ordered route row, sorted expert set, input tensor hash, weights hash, request, placement epoch, allocation epoch, and shadow lifetime. Do not merge Q32 resident capacity, top-10 width, one captured token, scheduler chunk size 32, or 32-slot bank capacity into a single `chunk_size` field.

## 7. Safe scaffold reuse and excluded broad matrix

`tests/layerwise_device_probe.py` is reusable as construction scaffolding, not as the workload. Reuse only its identity/manifest verification, physical-device checks, `OriginalExpertSource` loader, private template/backend construction, `cpu_reference`, exact shadow-weight check, and template-resident non-alias checks. The revised component probe uses one constructor lifecycle and, for A then B, directly executes the original backend sequence `execute_group -> synchronize -> release` with explicit `BufferIdentity` values. It does not claim the scheduler or whole-transport lifecycle.

Do not run its existing broad matrix: the all-layer source sweep; synthetic spread-resident initialization; preload of experts 100–131; 1/10/32-token by one-group/four-block cases; unused-slot hash probe; direct-abort/public-abort matrix; or extra repeats. Those are prior qualification scaffolds, not A/B, and would spend work outside the narrow decomposition question. The probe's prior scope is explicitly `LAYER0_SYNTHETIC_REAL_NPU_NOT_FULL_MODEL_OR_SCHEDULER`.

OFF and ON processes must each run exactly A then B through the same real provider lifecycle and resource configuration. Observation OFF must preserve the original full prepare, compute, synchronize, release, exception, and allocation behavior. Observation ON may record bounded Host subintervals and existing device completion evidence; it must not add a per-phase global synchronize, a new staging tensor, a retained source, or changed-slot behavior.

## 8. Component resource contract

This component fixture inherits the v20-style bounded component envelope, not the full-model v63 allocator envelope:

| Resource | Component contract |
|---|---:|
| NPU memory fraction | 4% |
| Derived allocator limit in prior v20 evidence | 2,631,841,873 bytes, about 2.45 GiB |
| Owned HBM guard | 4,096 MiB |
| Container Host memory / swap | 64 GiB / 64 GiB |
| Private 32-slot BF16 bank | 314,572,800 bytes |
| Revised 512 MiB-workspace reservation | 1,061,158,912 bytes |
| Historical 1 GiB broad-scaffold reservation (reference only) | 1,598,029,824 bytes |
| Prior observed v20 maximum board / owned process | 3,882 MiB / 1,060 MiB |

The 37,376 MiB allocator, 38,400 MiB owned-process limit, and 640 GiB Host limit recorded in `v20-admission.json` are full-model preflight fields embedded in reused admission material; they are not the component process's operational caps. The component launcher contract is the 4% allocator fraction, 4 GiB owned guard, and 64 GiB Host container limit. Physical 15, the existing shared-training allowance and board reserve, and fresh admission still apply.

## 9. Contract conclusion

The historical provider is sufficient for a minimal real-provider A/B decomposition without copying or reimplementing the prepare algorithm. The revised default-OFF observation changes in `TorchKTEPAdapter.copy_experts` and `OriginalExpertSource.read` sit beneath the already identified `AdapterObserver.copy_experts` parent and must preserve the two existing projection synchronizations. The component probe then uses the original backend's compute submission, synchronization, and release methods directly. Evidence may claim a captured decode-input-driven shadow **component** test and one derived replacement transition. It may not claim captured Prefill, scheduler coverage, whole-transport coverage, mixed-output equivalence, changed-slot reuse, full-model attribution, or performance qualification.
