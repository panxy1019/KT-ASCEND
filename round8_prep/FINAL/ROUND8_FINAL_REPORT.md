# Round8 final report — 2026-09-19

## Closure decision

**ROUND8 = COMPLETE_WITH_USER_SCOPE_CHANGE.**

The latest user explicitly cancelled further Round8 model performance testing
and requested closure after evidence audit. This report closes the revised
scope, not the original fully performance-qualified scope. No cancelled gate
is converted to PASS. No additional performance/cache-optimization campaign
is required or to be restarted automatically for Round8.

Selected handoff: **Qwen3.5-35B-A3B, Q32-C64 Static, Dynamic OFF, Deferred OFF**.
Q64-C64 remains the engineering quality reference. Q64 is not All-NPU.
Dynamic has system/product evidence, but no qualified performance benefit;
retain the implementation and fixes without enabling it by default.

## Evidence matrix

| Item | Final state | Evidence relative to round8_prep |
|---|---|---|
| Source/checkpoint and all40 CPU layer identities | PASS,30720 exact projection instances |10_HANDOFF/checkpoint-identity-v1.json; full-conversion-manifest-v1.json|
| CPU C64 operator integration | PASS in sampled scope |10_HANDOFF/CPU512_C64_SAMPLE_RESULT.md; MULTI_EXPERT_ENTRY.md; three-way-grouped-evidence-v1.tar.gz|
| Q64 full-model/system/restart | PASS in compact scope |10_HANDOFF/Q64_REFERENCE_AUDIT_AND_Q32_ENTRY.md|
| Q64 CMMLU reference |222/256,invalid0|10_HANDOFF/q64-quality-evidence-v1.tar.gz|
| Q32 structure/ownership/shared/system | PASS,40/40layers |10_HANDOFF/Q32_SYSTEM_FREEZE_AND_QUALITY_ENTRY.md|
| Q32 repeat/prefix/restart | PASS |10_HANDOFF/q32-system-evidence-v1.tar.gz|
| Q32 CMMLU |225/256,invalid0,+1.171875pp|10_HANDOFF/Q32_CMMLU_FINAL_AUDIT.md|
| Static bounded stability | PASS,2304tokens,two campaigns |11_DYNAMIC/STABILITY_AND_PERFORMANCE_AUDIT.md;raw/qwen-q32-stability-22-v2;raw/qwen-q32-stability-22-v3-resume|
| Dynamic migration/generations/weight identity | PASS in tested scope |11_DYNAMIC/NZ_TRANSFER_FIX.md;HELDOUT_V3_RELEASE_AUDIT.md|
| Dynamic held-out |128/128,8320updates,96triggered requests|11_DYNAMIC/qualification-release-v3.json|
| Dynamic CMMLU |224/256,invalid0,+0.78125pp;217/256triggered|11_DYNAMIC/CMMLU_V2_INDEPENDENT_AUDIT.md|
| Formal isolated performance | NOT_QUALIFIED_RESOURCE_AUDIT |11_DYNAMIC/PERFORMANCE_V3_RESOURCE_AUDIT.md|
| Further performance qualification | SKIPPED_BY_USER_SCOPE_CHANGE |latest user instruction and this report|
| Qwen Deferred | NOT_QUALIFIED,OFF |not inherited from DeepSeek|

Quality policy is the unchanged observed CMMLU256 delta >=-5pp versus Q64;
minimum210/256 for this reference. It is a product regression screen, not a
universal quality guarantee or statistical non-inferiority proof.

## Provenance and reproduction boundary

- Worktree:/home/admin/Documents/ChatGPT/KT/kt-round8-qwen35-prep.
- Branch:codex/round8-qwen35-prep; local commits only, no push/main/tag.
- Model revision:59d61f3ce65a6d9863b86d2e96597125219dc754.
- F32 bank manifest:35ca0ac63a3646ba7b89675053bbead19724325fa9d73f99910ef3c156a84405.
- Q32 mask:b358399e6405d60b81d8231529b1ec6024ff7138cddab2e0cbac3ccf0b25f486.
- CMMLU manifest:365f557c0746db1e02ddee4ec66c7afb5e72eeb17e8dd8e0a47d1ed2b7af1b9d.
- Dynamic candidate commit:9a183f309e1db794859646686431d9a1fbeb8819.
- Dynamic candidate SHA:69892f31dcbb450105a4adf153622c7bf6f5823292bfb8038ac9ae60160e55b9.
- Dynamic wrapper SHA:77dfbbab73431111e72b6ac2683e4f4ecfa617280436e9ce55a3d7f89177e46b.
- Runtime:.22 Kunpeng/910B3,NPU2,CPU128–191,NUMA4/5,C64=two32worker
  subpools,512intermediate split256+256,FP32partials; TP1,batch1,text-only.
- Exact imported Python/native/package/launch provenance is in archived resolved
  commands and frozen candidate configs; local gitlinks are NOT substituted for
  deployed overlays. Q32 mask remains32/256 per40layers, not32CPUexperts.
- Static and Dynamic are separately recorded deployments. For reconstruction,
  use their respective frozen resolved command, not an arbitrary current HEAD.

## Performance and memory limitations retained

All12Static/Dynamic ABBAcells were collected before cancellation. Descriptive
short/medium median E2E seconds:Static24.0788/47.2085,Dynamic24.3355/472.0047.
These are not isolated performance qualification or decode-throughput results.
Dynamic cold full-context preparation is included; no performance benefit claim.

The additional resource diagnostic confirmed owned swap0/OOM0 but substantial
NUMA4/5pressure,file reclaim and95532majorfaults. Per-wrapper BF16 cache is one
identified retention source; it does not explain all anonymous growth. This
investigation is retained as an unresolved Dynamic resource limitation, not
silently fixed or allowed to block the user-selected Static closure.
Static stability demonstrates a bounded plateau only for the tested workload.
It is not a zero-leak proof. Historical resource sampling limitations remain.

## Final audit performed at closure

Revalidated local SHA/readability for Q64quality,Q32system,Q32quality,and combined
Dynamic archives. Rechecked two stability summary reference trajectories exact,
18requests/campaign and2304total tokens; all40 conversion manifest entries sum
to30720exact projections. Re-ran frozen Dynamic auditor on .22:heldoutPASS128,
CMMLUPASS224/256,unchanged candidate/retry/config/artifact identities.
These are evidence re-audits, not new model executions. Historical failed
watchdog/admission/performance attempts remain archived unchanged.

## Local cleanup and handoff

Removed only612pip cached HTTP files(436.3MB) using pip cache purge. No installed
packages, source, uncommitted work, model, or evidence deleted. These download
caches can be regenerated, not restored as unique data. Free disk9MB→427MB.
Do not delete bundled runtimes or unrelated browser/user data to free more space.

Round9 may proceed through its own source/precision/resource/semantic gates.
This is authorization to continue the ordered plan, not Round9 device admission.
Qwen3.8 download remains incomplete; single-shard alternate transport also failed
at offset1409286144 with curl exit56. Preserve it and repair independently.
No Qwen3.6; no inherited Qwen3.8 Dynamic/Deferred/long-context qualification.
