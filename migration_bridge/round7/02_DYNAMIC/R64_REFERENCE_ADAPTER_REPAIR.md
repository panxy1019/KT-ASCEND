# Fresh R64 reference adapter repair

2026-09-17. This is a test/orchestration repair, not a production arithmetic
change or acceptance relaxation.

## Preserved failed attempt — A3_VERIFIED

`backing-requalification-v1` closed operators (100/100), Uniform compact
(32 requests plus four restart requests), and Frequency compact (32 plus four).
Both compact lanes passed same-path, prefix, restart, observer parity, finite,
mapping and ownership gates on the repaired private CPU runtime.

Its fresh R64 attempt completed CMMLU256 with175 correct and zero invalid, then
stopped at `FRESH_R64_REFERENCE_CHANGED`. No Dynamic DEV probe was started.
The175 result is not substituted for the frozen direct-reference174/256.

## Attribution — CODE_INSPECTED

Historical `/workspace/results/r65-r67-serial-v5/R64/quality/resolved-command.json`
contains a direct Ascend All-NPU command, without any `--kt-*` arguments.
The new adapter incorrectly launched KTEP with64 resident experts and expanded
context/prefill limits. This was not the same reference execution contract.

The repair strips the KT suffix only for the explicit `reference_r64` lane and
restores historical context2048, max-total4096, chunk2048 and max-prefill2048.
Other lanes are unchanged. Reference Dynamic/Deferred/observer/recorder features
are forbidden. The resumed driver also verifies the resolved command against
the original command and still requires174 correct, zero invalid.

## Verification and recovery

LOCAL_VERIFIED: all82 pure-Python tests pass, including four reference-adapter
regressions; diff check passes. This does not establish A3 reproduction.
Resume requires the preserved exact failure and all three actual predecessor
PASS summaries, binds their SHA256 values, and uses a new result directory.
No old result is overwritten or reissued as evidence at a new tool revision.

A3 lossless archive of both post-fix compact lanes and the failed R64 adapter:
`post-fix-static-and-r64-adapter-failure-v1.tar.gz`, SHA256
`b0a4be94684d862ab553affde27506de745c978d733d53f5be2a046e2de616bb`.
Local transfer verification is a separate step before claiming local preservation.

Next: corrected direct R64 reproduction, then DEV-only migration probes
256/512/1024. No threshold fit, held-out evaluation or quality-rule change occurs
before this predecessor closes.
