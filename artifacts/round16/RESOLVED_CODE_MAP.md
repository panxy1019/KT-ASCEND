# Round16 resolved code map

This map is a DEV reference. It does not claim S15 compatibility or device
qualification. All listed Round14a files were read in place from the Round15
worktree without modification; only these small files were hashed.

## Frozen Round14a parent

| Role | Read-only path under `artifacts/round14a/a-gdn-model-snapshots/v295` | SHA-256 | Relevant boundary |
|---|---|---|---|
| Qwen model owner | `overlay-python/sglang/srt/models/qwen4_exp.py` | `1986d8a880fe1503848eadf02a8ed572d543eac967fcb583b6f67d42250f3545` | `Qwen4ReferenceGatedDeltaNet`, `Qwen4ReferenceIndexer`, `Qwen4ReferencePLE`, decoder/model forward |
| GDN primitive | `overlay-python/sglang/srt/layers/round9/gdn.py` | `65d990b19b6cfb35ea8f86e94985120a4a37c8397f234173365ec8655f25ebd1` | `sigmoid_gated_rms_norm`; local sigmoid semantics stay frozen |
| QSA primitive | `overlay-python/sglang/srt/layers/round9/qsa.py` | `0fd8d98b37f7e90e90f9866356892236dcc7d0114351d1438025561df6c1bd35` | `select_tokens`; causal/signed ranking/GQA order stay frozen |
| PLE async runtime | `overlay-python/sglang/srt/layers/round9/ple_async.py` | `792d077c8a1b2e302813eec05d25be0446f9b86dcb1faaf2e3e76f8499c11280` | existing ticket and ready/consumed owner |
| PLE binding | `overlay-python/sglang/srt/layers/round9/ple_binding.py` | `77b1db726731286fda53d8a49572aded19a9fc2110a4853856b5f3bdc2040e5b` | state binding; no second history owner |
| KT Expert wrapper | `overlay-python/sglang/srt/layers/moe/kt_ep_wrapper.py` | `77dfbbab73431111e72b6ac2683e4f4ecfa617280436e9ce55a3d7f89177e46b` | CPU submit/wait, resident `gpu_method.apply`, merge and lifecycle |
| PLE state pool | `overlay-python/sglang/srt/mem_cache/ple_state_pool.py` | `110a33c09c5b2d2699920e4d278d4081d76d6bcd087974608caf22666e297d9a` | SGLang remains the state owner in A |
| QSA state pool | `overlay-python/sglang/srt/mem_cache/round9_qsa_pool.py` | `99ae4f5a3db4af3c1c3fe1e79295dd0c2159fa5427271678b605d7d09c567a7a` | SGLang remains the state owner in A |

## Round15 read-only dependency evidence

`artifacts/round15/PROVIDER_MAP.md` currently classifies:

- QSA `npu_lightning_indexer` as a pending component with a PA layout gap;
- GDN `npu_rms_norm` as a pending component that must retain local sigmoid;
- resident BF16 grouped MoE as the already-current provider rather than a new
  speed candidate;
- segmented graph scope as the resident NPU `gpu_method.apply()` region only.

`artifacts/round15/OFFLINE_CAPABILITY_RECEIPT.json` records torch 2.9.0,
torch-npu 2.9.0.post2 and CANN 9.0.0 in the frozen KT image, with device-free
torch-npu import still unqualified. Round15 remains in P0 repair and none of
those pending paths is S15 yet.

## Ownership boundary

Round16-A may adapt explicit metadata and call an injected provider, but the
SGLang scheduler, CPUInfer and request state pools retain ownership. Round16-B
may adapt the vLLM model's MoE call to the same expert core, while vLLM retains
scheduler, KV/QSA/GDN/PLE allocators and sampler ownership. The common contract
contains identities and leases only; it contains no runtime or device import.

The A and B agents append their exact module and symbol maps in separate
`RESOLVED_CODE_MAP_A.md` and `RESOLVED_CODE_MAP_B.md` receipts. S15 changes are
resolved once, after the terminal barrier, and produce a new compatibility
identity rather than relabeling these Round14a files.
