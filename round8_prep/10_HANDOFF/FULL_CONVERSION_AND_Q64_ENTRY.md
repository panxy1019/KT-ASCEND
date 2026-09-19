# Full conversion → Q64 bring-up, 2026-09-19

## Conversion closed (supersedes running status below)

All40 layers/120 GGUF tensors/30,720 per-expert projections passed exact source
comparison; payload128,849,018,880 bytes (120GiB). Raw final manifest frozen as
`full-conversion-manifest-v1.json`, SHA256
`35ca0ac63a3646ba7b89675053bbead19724325fa9d73f99910ef3c156a84405`.
Actual launch-v2 source/native1688 hashes and command captured in
`q64-resolved-command-v2.json`, SHA256
`b0b62f15266a8512dd05fd3e5698969ecf216a570c25d7db09b483538fc9abd8`.
Freeze these before first full-model launch. No full-model PASS yet.

User explicitly authorized continuous operator migration, middle/late probes,
all40-layer conversion/full identity audit, then Q64-C64 static bring-up.
Pause only for long quality/performance campaigns, not ordinary software fixes.

## Operator prerequisite closed

Layer0/20/39 each used all256 real weights in GGUF and fixed eight logical IDs,
qlen1/8/64, ten repeats each. All finite/repeat-exact; all GGUF projections
roundtrip exact. Maximum relative-L2 respectively0.001636731/0.001624075/
0.001644756, under0.01. Raw reports/logs archived in
`multi-expert-results-v1.tar.gz`. This is synthetic-operand CPU local evidence,
not full-model/router/shared/accelerator equivalence qualification.

## Full conversion running

`tools/convert_all_experts.py` requires all three successful reports and full
checkpoint identity; rehashes source files, writes one3GiB layer GGUF at a time,
then reopens source BF16 tensors and checks every expert/projection value.
Expected40 layers,120 tensors,30,720 projection instances,120GiB matrix payload.
MTP and vision excluded. Output `/workspace/round8/cpu-experts-f32-v1`;
manifest only reaches final PASS after all layers. Driver3088384 on .22,
host log `convert-all-v1.log`; preserve healthy conversion.

## Runtime provenance and static launch

Copied Round7 final deployed `runtime-v8/sglang-python` into private Round8
`/workspace/round8/sglang-python`. Old source remains read-only. Native KT is
the previously hash-verified private package with C64 two32 subpools. This is
an explicit deployed-artifact lineage, NOT a claim the local gitlink equals
the deployed source. Full actual Python/native hashes go into resolved command.

`prepare_q64_launch.py` freezes bit-reversal uniform64/256 per40 layers,
and proves the official mask loader reproduces it exactly. The CLI frequency
loader is only transport for synthetic binary mask scores, NOT learned frequency.
No CMMLU data used. New launch-v2 adds test-only structure/routing hooks.

Static NPU BF16/unquantized, CPU LLAMAFILE F32, TP1,batch1,context4096,
chunked prefill512, max total tokens4096. Graph/Deferred/Dynamic/speculative/MTP
off, text-only, shared fusion off, radix off. Actual parsed args must still be
audited after server start. Q64 is not All-NPU reference.

Before launch: freeze conversion manifest and source/command artifacts in local
commit; run fresh host admission using `admit_q64_host.py`, then `start_q64.py`
within60s. Memory384GiB/no extra swap; see memory admission amendment. NPU2,
CPU128–191,NUMA4/5 unchanged. No other workload may be stopped.

Ordinary launch failures: preserve log, reproduce, inspect current then official
source, minimal private adapter, test/retry. Do not claim model PASS on import
or operator evidence alone. No push/main merge/tag.
