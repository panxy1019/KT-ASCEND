# Round16-A resolved code map

Status: `PASS_DEV_CPU_MOCK_DEVICE_REQUIRED_NOT_RUN`

This map resolves the A0-A3 provider boundary. It does not copy or reimplement QSA, GDN, PLE, or MoE kernels. The installable `round16.a` package imports only the standard library and the shared `round16-contract-v1`; SGLang, KTransformers, torch, torch_npu, CANN, and an Ascend executor must be injected later.

## A0 — provider and lifecycle boundary

| Concern | Round16-A symbol | Contract behavior |
|---|---|---|
| Manifest and dependency classification | `round16.a.provider_manifest.manifest_payload`, `dependency_tiers` | P0 is required and CPU-testable; P1 is optional framework wiring; P2 is device execution and remains `DEVICE_REQUIRED_NOT_RUN`. |
| Static support decision | `round16.a.provider_manifest.decide_support` | Checks operation, phase, metadata version, state format, BF16, layout, device identity, stream identity, and injected-backend availability. |
| Metadata adaptation | `round16.a.sglang_adapter.adapt_metadata` | Accepts a mapping or attribute object, reconstructs a tuple-backed `ProviderSpec` snapshot, and matches all seven `StateLease.identity` fields before execution. |
| Runtime input observation | `round16.a.sglang_adapter._observe_inputs`, `_match_observed_inputs` | Mapping keys are logical names; sequence/single inputs use positional spec names. Tensor objects only need shape/dtype/layout/stride/device. Count, resolved names, and every physical field must equal the frozen spec before support or fallback selection. |
| Runtime stream observation | independent metadata field `runtime_stream_kind` | Must equal `ProviderSpec.stream_kind`; a forged nested spec cannot establish runtime stream identity by itself. |
| Execution and fallback | `round16.a.sglang_adapter.KTAscendProvider.run` | Unsupported input takes the injected fallback branch before the backend can run or write state. |
| Completion / lease / writes | shared `ProviderResult` plus `KTAscendProvider._validate_result_lease` | Completion ID and lease identity are preserved; unleased state writes are rejected. |
| Shape envelope | `round16.a.provider_manifest.decide_support` | Accepts exactly one `hidden_states` tensor with rank 2, shape `[1..128, 2048]`, BF16, ND, and omitted or `(2048, 1)` stride. Unresolved subshapes reject before execution. |
| Alias | `BackendResult.aliases_input` and `_aliases_runtime_input` | A backend or fallback may not return the runtime input container or any top-level mapping/sequence member by identity, even if the backend fails to declare the alias. Framework-specific storage overlap remains a device gate. |
| Lifecycle | `KTAscendProvider.close` | Immediately rejects new work, attempts every dependency despite close failures, aggregates errors, retries failed resources on later calls, and never recloses a successful resource. |
| Registration | `round16.a.registration.register_provider` | Explicit, idempotent, conflict-detecting; import performs no registration. |

## A1 — QSA Lightning Indexer boundary

- Provider operation: `qsa.lightning_indexer`, Decode only, state `qsa-index-v1`.
- Frozen source: `/home/admin/Documents/ChatGPT/KT/kt-round9-mainline/artifacts/round14/a-snapshots/v235/overlay-python/sglang/srt/layers/round9/qsa.py`.
- Frozen symbols: `compress_index_keys`, `select_tokens`, `sparse_gqa_candidate`.
- SHA-256: `0fd8d98b37f7e90e90f9866356892236dcc7d0114351d1438025561df6c1bd35`.
- Fallback: eager `select_tokens()` and the promoted sparse-GQA choice. Layout rejection occurs before state update; no PA cache reinterpretation is performed by this component.
- Graph capability: not advertised. Device schema, layout construction, numerical agreement, and capture remain `DEVICE_REQUIRED_NOT_RUN` / `NOT_RUN`.
- Round15 boundary: frozen terminal `ROUND15_COMPLETE_SIMPLIFIED_DIRECTIONAL_NO_PROMOTION`. No S15 promotion is inferred from source/API presence.

## A2 — GDN RMSNorm plus local sigmoid gate boundary

- Provider operation: `gdn.rmsnorm_sigmoid_gate`, Prefill and Decode, state `gdn-state-v1`.
- Frozen source: `/home/admin/Documents/ChatGPT/KT/kt-round9-mainline/artifacts/round14/a-snapshots/v235/overlay-python/sglang/srt/layers/round9/gdn.py`.
- Frozen symbols: `gated_delta_recurrence`, `sigmoid_gated_rms_norm`, `eager_gdn`.
- SHA-256: `65d990b19b6cfb35ea8f86e94985120a4a37c8397f234173365ec8655f25ebd1`.
- Model call point evidence: `Qwen3GatedDeltaNet.forward` in `/home/admin/Documents/ChatGPT/KT/kt-round9-mainline/artifacts/round14/a-snapshots/v235/overlay-python/sglang/srt/models/qwen3_next.py`, SHA-256 `d0bbb7b6f5ea4a54b32a47413855deca09113008e32f28f831995938478a6ee5`.
- Fallback: the frozen recurrence/core and eager `sigmoid_gated_rms_norm`; the provider does not alter recurrent or convolution state.
- Graph capability: not advertised. Native RMSNorm dispatch, numerical agreement, full-parent speed, and capture remain `DEVICE_REQUIRED_NOT_RUN` / `NOT_RUN`.
- Round15 boundary: frozen terminal `ROUND15_COMPLETE_SIMPLIFIED_DIRECTIONAL_NO_PROMOTION`; no request or S15 gate promotion is claimed.

## A3 — resident grouped-MoE boundary

- Provider operation: `moe.resident_grouped`, Prefill and Decode, state `resident-moe-v1`.
- Frozen wrapper source: `/home/admin/Documents/ChatGPT/KT/kt-round9-mainline/artifacts/round14/a-snapshots/v235/overlay-python/sglang/srt/layers/moe/kt_ep_wrapper.py`.
- Frozen symbols: `KTEPWrapperMethod._submit_cpu_forward`, `_sync_cpu_forward`, `apply`, and `SharedFullContext.close`.
- SHA-256: `77dfbbab73431111e72b6ac2683e4f4ecfa617280436e9ce55a3d7f89177e46b`.
- Responsibility split: the candidate boundary is the existing resident `gpu_method.apply()` region. CPU submission/wait, PLE host work, route production, cancellation, CPU/NPU merge, and later layers stay outside.
- Fallback: the same eager grouped-MoE provider. This is `NO_NEW_A_CANDIDATE_ALREADY_CURRENT_PROVIDER`, not an A speedup candidate.
- Graph capability: provider identity is stable, but capture qualification is a later B/device gate and is `NOT_RUN` here.

## PLE and scheduler ownership

`ple_model_ready`, `ple_before`, and `ple_after` in `/home/admin/Documents/ChatGPT/KT/kt-round9-mainline/artifacts/round14/a-snapshots/v235/overlay-python/sglang/srt/layers/round9/ple_binding.py` (SHA-256 `77b1db726731286fda53d8a49572aded19a9fc2110a4853856b5f3bdc2040e5b`) show that PLE owns scheduler/request lifetime, host budgeting, and device-gated ring setup. Those responsibilities are deliberately outside the A1-A3 provider executor.

## Round15 evidence and responsibility boundary

- `PROVIDER_MAP.md`: SHA-256 `823b269e68d3615a6669278ce59629eb07d4af2e7be52e0f192c4ae718af6ced`.
- `OFFLINE_CAPABILITY_RECEIPT.json`: SHA-256 `dd9969758b8949590596dae136e030b58990ab8d162e5ad6a4ad091ad26ebf56`.
- These documents establish operator-source availability and intended fallback/ownership. They do not establish device import, numerical equivalence, request behavior, performance, or graph eligibility.
- The provider map points to a later frozen parent snapshot (`v295`) with a different Qwen source hash. This Round16-A map records the directly inspected, user-authorized `v235/overlay-python` files and hashes; it does not equate the two snapshots.

## Fresh A1 real-start and attribution chain

- Trusted build donor: local read-only `/home/admin/Documents/ChatGPT/KT/kt-round9-mainline/artifacts/round14a/a-gdn-model-snapshots/v295`. The actual remote parent is `/mnt/data1/kt-round9/round14a-a-gdn-model-20260927-v2/v295`; its ten non-budget identities match, but its historical execution budget begins with SHA-256 `046fc30c` instead of the required unspent `665ded641f8848d65ca44776cf74dfd70909184339a619f3d70befee09c36974`, so it is evidence only and cannot be a build donor. A/B must be built locally, frozen under externally reviewed manifest/budget/ledger hashes, then transported to new remote paths.
- Builder/controller: `round16.a.fresh_a1_snapshot`; the copied launcher uses distinct A/B container and evidence identities, a one-start budget, the exact six-request queue, profile env `ON`, and zero retries.
- Request driver: `round16.a.fresh_a1_device.driver`; it reuses v295 `round11_t4_driver.post/consume_stream` (SHA-256 `38faa29ce4583254c59ee32bdf921329cf97aa0b04393a3a4bc5c9ef2d1ca2be`) and `round10_t1_protocol.validate_response` (SHA-256 `3b5d96864922ad4bd51d298ebefd039fe06e2de9a1bef8de4f800c8dcf2e0129`).
- Host lifecycle: `tools/round16_a1_remote_block.py`; it reuses v295 launch/guard, `m3_cleanup.wait_for_guard_exit` (SHA-256 `b6df7cca2b2ac27d29acbd75515c58610372f0d20a13f050be11edfd7161a201`), and `candidate_resources.sample` (SHA-256 `8adbeadc39665fffd495a6ccd275bf7448c99414352cff531d27a5541ec012e9`).
- The frozen Round14a v298/v299 profile chain is the attribution reference: `source/tools/launch_hybrid.py` SHA-256 `6eae519ebb64a4c0c57220d2f58e5fcdf62282987a168effaa62392260478ecc`, `round14a_b_profile_driver.py` SHA-256 `325f9e06901cca49c4f92ca4f3fab9112d3500343af7864b297972f9c1d01e44`, `round14a_b_profile_supervisor.py` SHA-256 `8cc6bc71a28479e88fa212503aea2fc928e2efdb2e3c364cf1dab7443a6b9a89`, and `round14a_b_profile_preflight.py` SHA-256 `8547c9886b1d0362db8498fc049e319c0c9b36acec5413cd453642aea54b1c91`.
- Runtime binding follows the real ownership path `module.quant_method -> KTEPWrapperMethod`. Host parent/dispatch/sync/merge and CPU UNKNOWN cover layers 0..47 exactly; direct NPU expert events are sampled at 0/24/47. Not-ready NPU events remain in the bounded cross-batch queue and are queried later without synchronization.
- `round16.a.fresh_a1_evidence` verifies both binding receipts, copies complete Round13 QSA_INDEXER, GDN, and NPU_RESIDENT event measurements for W1/W2 only, requires a serialized post-W2 `SHORT_DRAIN` batch as the FIFO flush witness, and creates the Decode bounding interval from direct serial scheduler boundaries. The final canary may remain pending. The copied launcher binds the verified observer to `/out/round13-spans.jsonl` and `/out/round13-bindings.json`; the separate A1 observer supplies Host layers 0..47. The hook map is a boundary catalog, not an installation list; A1 does not install `NPU_SHARED`. When no isolated A2/A3 interval exists, `analysis_payload` preserves `A1_INSUFFICIENT_ATTRIBUTION`, candidate null, and A2/A3 evidence gaps.
- Admission must be regenerated immediately before each A/B block. Host control rehashes the exact result and its two wrapper-named adjacent raw >=60-second rounds, verifies their order/envelope, card6/chip0/physical12/PCI `0000:85:00.0`/NUMA `[2,3,4,7]`, and <=600-second raw/result freshness. The result and raw files are copied into evidence under their reviewed SHAs; archive file selection comes from the reviewed wrapper rather than a historical RESUME name.
- The exact reviewed `round16-a1-source-sha256.json` SHA is required at host prelaunch and again inside the container driver. Immutable source excludes `execution-budget.json`; that file is separately constrained to initial-unspent then exactly one matching full-ID model start, with both SHA identities recorded.
- Cleanup binds terminal ACKs to the full ID and handles a never-started `created` container only after PID-zero verification and bound-guard termination or confirmed prior exit. Archive evidence includes SHA-256 and size for server, guard, and driver logs. Block B additionally requires the exact ordered six A request IDs plus A's ledger-after SHA, PASS supervisor cleanup state, and archive binding before the first B subprocess.

## Explicitly unrun

Ascend import, torch_npu dispatcher calls, complete-model execution, device numerical checks, graph capture/replay, CPU benchmarks, performance gates, and full bank reads are all `NOT_RUN`. No NPU, SSH, shared object, global environment, or top-level KTransformers package change was made.
