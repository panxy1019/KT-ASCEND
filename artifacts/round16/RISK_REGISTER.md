# Round16 DEV risk register

This register separates known integration work from device or performance
claims that cannot be made before the Round15 barrier.

| Risk | Current evidence | DEV treatment | Post-barrier gate |
|---|---|---|---|
| S15 interface drift | Round15 is in P0 census repair, so S15 does not exist yet | Bind only to `round16-contract-v2` and Round14a read-only symbols | Re-resolve S15 source/dependency tuple and rerun affected tests |
| KT/donor ABI mismatch | KT uses torch 2.9/torch-npu 2.9/CANN 9.0; donor source came from torch 2.10/torch-npu 2.10/CANN 9.1 | No donor `.so` or runtime import; record source algorithms and schemas only | Rebuild against the selected runtime; verify dispatcher schemas and loaded `.so` hashes |
| Device-free torch-npu import | Round15 bounded preflight could not qualify it without the device image | Lazy P2 import, explicit `DEVICE_REQUIRED_NOT_RUN` | First admitted component process performs import/schema preflight before allocation |
| QSA layout incompatibility | Frozen KT index pool and donor PA cache layouts differ | No reinterpretation or guessed stride; unsupported layout falls back before writes | Real fixture proves metadata, page table and causal expansion |
| vLLM extension API mismatch | Donor commit exposes model registry APIs; local vLLM 0.26 runtime has not been loaded | Unverified registration returns `VERSION_API_UNRESOLVED` without touching the registry; only a `VerifiedModelBinding` can register the CPU-tested dynamic wrapper | Resolve constructor/call/worker APIs in the selected image before creating the verified binding or allocating the model |
| Full-expert temporary NPU allocation | Replacing forward alone would allocate 512 BF16 experts per layer | Selective-loader plan keeps logical 512 and physical Q32, with alias/rollback tests | Allocation tracker proves no full-expert temporary and measures peak HBM |
| State ownership and late completion | vLLM owns KV/QSA/GDN/PLE slots while CPU work completes asynchronously | Full request/allocation/batch/layer/token/completion generation identity | Cancel/reuse smoke and one-layer state reference on device |
| Graph replay reuses captured CPU result | CPU submission is outside a legal device graph | Graph-boundary mock requires an active lease, non-empty input fingerprint, monotonic replay epoch and fresh CPU task; NPU/submit exceptions retire reservations | Two distinct real inputs produce two CPU submissions/results; external worker cancellation/idempotency still needs device integration proof |
| Mutable metadata expands authority | Frozen dataclasses previously retained caller-owned lists during review | `round16-contract-v2` deep-freezes tensor, route, lease, provider and write sequences and rejects bool/non-integer identities | Rerun the same contract tests after the S15 rebase; a future semantic change requires v3 |
| Carrier package cross-install | Early installers copied the whole Round16 tree | A installs only common+A; B installs only common+B; isolated tests prove the other carrier is absent | Preserve carrier-scoped package lists in any later wheel/build metadata |
| Local disk headroom | Local `/dev/sda1` had 2.7 GiB free after sparse checkout | Three sparse ~60 MiB worktrees; no full checkout, model or bank copy | Preserve sparse layout; device evidence remains on registered remote storage |
| Resource interference | Round15 uses the same cluster and its lease is not released | DEV performs no remote or device work | Terminal receipt, fresh admission, and user-authorized alternate safe card/NUMA selection |
