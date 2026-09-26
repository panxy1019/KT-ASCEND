# Round10 append-only execution ledger

## 2026-09-26 — branch and T0 recovery

- Created branch `codex/round10-hybrid-kunpeng-dynamic` in the existing linked
  Round9 worktree; base HEAD remained
  `b41e901c059222b8c1e4367abe193d7f58cc1d83`.
- Preserved the pre-existing dirty worktree in `INITIAL_WORKTREE_STATUS.txt`.
- Found the exact remote v90 snapshot at
  `/mnt/data1/kt-round9/plan15-threshold-candidate129-20260925-v90` and copied it
  to `h0-v90-source/`.
- Verified the snapshot's 2,560-file SHA-256 manifest with no missing or changed
  file. The mutable controller/budget files excluded by the original manifest
  remain evidence, not H0 source inputs.
- Recorded model and tokenizer identities in `T0_MODEL_IDENTITY.json`.
- Ran one CPU-only, network-disabled provider identity probe in the v90 image;
  it used no NPU device and was removed. Results are in
  `T0_PROVIDER_IDENTITY.json`.
- Read-only remote inspection found no physical-12 process and no owned task
  process. External `qwen38-vllm-*` and training containers remain untouched.
- Device/model budget consumed by T0: zero starts, zero generations, zero
  abort controls.

This file is append-only. Corrections are added as later entries rather than
rewriting the historical observation.

## 2026-09-26 — T1 admission and first model process

- The first admission validator rejected its own raw samples because its NUMA
  parser treated `Node 2 MemFree:` as ordinary `/proc/meminfo`. The failed
  evidence was preserved. A failing parser test was added, the parser was
  repaired, and a new two-round `admission-r2` run passed all 12 windows.
- v91 created one owned container but the inherited guard allowed only
  `kt-round9-*` names. Docker never started it; `model-start.json` was absent
  and the snapshot budget stayed at zero. The exact ID was verified in
  `Created`, `Pid=0`, never-started state while physical12 was empty, recorded
  as `OWNED_NEVER_STARTED_VERIFIED`, and removed.
- The guard whitelist was narrowed to the additional exact pattern
  `kt-round10-t1-[abr]-v[0-9]+`. v93 then started under the same H0 payload.
- v93 completed four generations. Marker, 391, and JSON passed their answer,
  token/logprob, path, and provider contracts. The fourth quote canary failed
  its exact answer contract. The old driver did not preserve the failing raw
  response before validation; this is recorded as an evidence gap. The model
  process was stopped, guard acknowledged exit, and the container was removed.
- T1 consumed 1 model start, 4 generations, and 0 abort controls. The frozen
  v95 recovery schedule uses the remaining start for 19 generations and two
  abort controls, keeping final use within 2/24/2.

## 2026-09-26 — T1 recovery, qualification, and cleanup

- A targeted file-cache retirement first failed before `fadvise`: the CPU-only
  container could not audit all host `/proc/*/maps`. The same fail-closed script
  was then run on the host as root. It found no active references to the exact
  179 MXFP4/model files, used only `POSIX_FADV_DONTNEED`, preserved every file
  identity, and restored NUMA7 MemFree from 28.74 to 224.28 GiB. No global
  cache operation or external process change occurred.
- A wholly new two-round v95 admission passed 12/12 windows. Minimum
  conservative SMT-sibling idle was 99.578%; minimum NUMA free memory was
  230.34/229.91/156.87/224.28 GiB for nodes 2/3/4/7. The complete training
  container ID/cgroup remained stable and physical12 had no unknown process.
- v95 used the second and final T1 model start. Its single driver completed the
  frozen 19-generation recovery schedule and two cancellation controls with
  zero retry. It returned `PASS_ROUND10_T1_PROCESS`.
- Across v93 and v95, all five canary categories passed. Live
  63/64/65/127/128-token edges used Hybrid and returned ACK/EOS. Decode cases
  produced 32 tokens to the length limit and 49 tokens to EOS with finite
  log-probabilities. Both mid-prefill cancellation lifecycles had no late
  commit, returned the lease, and passed same-process reuse.
- Supervisor cleanup obtained guard exit ACK, removed the exact owned
  container, and left physical12 empty. T1 final budget is 2 model starts,
  23/24 generations, 2 cancellation controls, and zero automatic retries.
- T1 is frozen as `PASS_T1_H0_ROUND10_NARROW`. Live 129/130 requests were not
  spent; those lengths remain outside the qualified Hybrid range.

## 2026-09-26 — T2 Hybrid cost attribution and fixture

- Reused the v95 host-only observer, which adds no device synchronization. It
  contains 39 real Hybrid chunk selections and 78 complete sampled applies at
  layers 0 and 3.
- At 32 tokens the sampled enclosing apply median was 3.430 ms and CPU result
  wait median was 1.311 ms. CPU submit and resident NPU apply medians were
  host-return intervals of 0.302 and 0.394 ms. They are nested clocks and are
  neither summed nor extrapolated across all layers.
- Preserved a layer0 cold outlier of 868.612 ms. It is not used as steady-state
  performance. Decode has request-level E2E and token continuity evidence but
  no synchronized per-stage device attribution.
- Archived the immutable real hidden/route/reference fixture from v80. The
  tensor SHA-256 is c837641468b9db93c422a3610af7a7ecc1f14d4bbb359841d69ac0dfd655dc6e
  and the complete route-tape SHA-256 is
  201a9b65b24c2fd9774c8c2dbfc4f795078d12dc8f20ce4b2a82efee12d857c0.
- T2 consumed zero additional model starts and zero generations.

## 2026-09-26 — T3 current Kunpeng native path

- Six bounded component identities were used. v96-v99 preserved probe launch
  contract failures (package root, relative import, pinned-provider device,
  and NUMA capability). No forward or result was silently retried.
- v100/v101 reached the current `KTMoEWrapper -> CPUInfer -> llamafile MOE`
  path with 4x32 workers on NUMA 2/3/4/7 and four 160-wide intermediate
  partitions.
- Two real layer0 groups matched an independent MXFP4 decode reference at
  relative L2 3.27e-7 and 2.00e-7 with exact repeat hashes. Steady medians were
  1.468 and 1.465 ms.
- The MXFP4 oracle differed from the captured BF16 contribution by 0.1290 and
  0.1280, failing the pre-run exploratory 0.05 cross-precision gate. The raw
  FAIL is preserved and no BF16-equivalence claim is made.
- No CPU performance candidate was promoted. The H0 C128/MXFP4 backend remains
  enabled; physical12 was empty after v101. T4 proceeds independently.

## 2026-09-26 — T4 Dynamic disposition

- One v102 model start covered five generations and one abort. Four Hybrid prefill-end commits preserved live KV/GDN/QSA/PLE state; cancellation published no placement and same-process reuse returned `391`.
- H0 median was 5.188200s; D1 median was 23.648729s (4.558176×). D1 remains default OFF.
- Guard ACK, container removal and physical12 cleanup passed.

## 2026-09-26 — T5 GDN/QSA/PLE and state qualification

- v103 GDN and QSA independent references passed. GDN max FP32/BF16 error was 1.08750e-6/0.00506596; QSA was 1.07935e-6/0.00511660. Repeat, index, signed Top-K, decode/reset and allocator retirement gates passed.
- v103 preserved a physical-device gate failure; v104 preserved a literal `{rev}` controller-path failure. v105 PLE itself passed all eight exactness/ownership/lifecycle checks and three byte-exact cases, then the old offline aggregator rejected the real GDN key name.
- The aggregator schema was repaired test-first and v103 GDN/QSA plus v105 PLE were combined offline with fixed input hashes. T5 used 3/6 component processes, no model start, and no B performance candidate.

## 2026-09-26 — T6 final integration and retirement

- Fresh 12-window admission passed; minimum conservative idle was 99.6236% and all resource/identity checks passed.
- v107 completed 8 generations and one abort. All outputs match corresponding v95 tokens; Decode reached 32 steps; cancel/reuse passed. Unprofiled H64 median was 5.621022s, 1.083424× v95 H0.
- The v107 supervisor is PASS with driver exit0, guard ACK and container removal. The outer controller then failed on the old T1 receipt filename; raw evidence is retained, the generator is fixed, and no second model start was used.
- Targeted cache retirement audited and retired 179 unreferenced model/bank files with POSIX_FADV_DONTNEED. Physical12 has no process and automation remains paused.
- Final status is `ROUND10_AB_NARROW_QUALIFIED`; no speedup is claimed and no candidate is promoted.
