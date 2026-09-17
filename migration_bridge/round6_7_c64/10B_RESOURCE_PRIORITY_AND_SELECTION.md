# Resource priority and final comparison — 2026-09-17

User instruction: finish R16-C32; stop R8-C32 with
`STOPPED_BY_RESOURCE_PRIORITY`; continue the active C64 chain unchanged:
performance → memory audit → R8-C64 smoke → R16-C64 CMMLU256 → R8-C64 CMMLU256.

The independent R8-C32 container is explicitly stopped, not deleted. Its raw
results remain historical partial evidence (91/256, 64 correct at stop request),
not a quality FAIL and not usable as a complete R8 quality comparator.
See RESOURCE_PRIORITY_20260917.json. R16-C32 was 252/256, 171 correct.

Performance workload contention changes at this stop event. Report session
timestamps and this intervention; the existing run remains exploratory,
never silently converted to an exclusive-host benchmark.

## Required final comparisons

- R16-C64 versus completed R16-C32: item-aligned correctness pairs and delta.
- R8-C64 versus fresh Round6.7 R64: identical manifest and evaluator, full256,
  invalid count and -5pp gate (174 baseline requires at least162 correct).
- R8-C64 versus R16-C64: item-aligned correctness, capacity and measured timing.
- Validate source/native/model/tokenizer/profile/manifest/evaluator provenance;
  label expected CPU-backend changes rather than claiming identical arithmetic.

## Selection

1. R8-C64: system PASS, full quality within5pp of R64, and usable performance
   supports selection as heavy CPU baseline.
2. If R8 quality fails, R16-C64 may be selected only if its own system and
   full quality gates pass.
3. If R16 also fails, evaluate R32-C64 or R48-C64 as new frozen candidates,
   not as already-qualified fallbacks.

No absolute usable-throughput requirement has been supplied. Report measured
latency/throughput and a clearly labeled recommendation; do not invent a hard
threshold or equate this prefill1-token test with decode performance.
The existing background chain executes through quality; final cross-profile
comparison/selection requires review after complete artifacts exist.
