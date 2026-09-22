# Stateful decode prequalification — interim A3 results

This is engineering evidence, not QD3/HD3 numerical approval.

## Small controls

| Check | Result |
| --- | --- |
| Pure and CPU-tensor adapter tests on A3 | 13/13 PASS |
| Natural defer0, 4 x8 targets | 36 complete forwards PASS |
| Fresh defer0 self-history forcing | 36/36 full-logit byte exact |
| Uniform synthetic forcing | 32/32 non-greedy injections, next-input checks PASS |
| Defer1 self/synthetic histories | 72 complete forward associations/transactions PASS |
| Final physical/index audit | 36 baseline +72 force0 +72 force1, unknown=0 |

These use independent processes/caches. Synthetic controls are explicitly
excluded from pilot/qualification exposure and numerical threshold fitting.

## Long defer0 full/compact comparison

20 requests, 64 targets each: 1280 targets +20 production-overlap lookahead.
All1300 indexed/physical forwards audited after planned shutdown, no unknown or
incomplete captures. Baseline four prompt families repeated5 times each had one
unique64-token trajectory per family. The old8-token smoke output was an exact
prefix for all4 matching baseline prompts.

Full natural baseline versus fresh compact forced self-history:

- Full-logit and retained tensor bytes: 34045 tensor comparisons across1300
  forwards PASS. Dropped full activation tensors are not claimed as validated
  in compact mode, except the preregistered full-diagnostic sequences.
- Full raw capture bytes: 1380241180.
- Compact raw capture bytes: 592963662 (about57% reduction).
- Approximate observed request time: full10–12 seconds, compact8–9 seconds
  per64-token request. This is sequential engineering telemetry, not a controlled
  production throughput claim or a batch qualification.

Evidence: `evidence/long-baseline/` and `evidence/long-force0-compact/`.

## Still required

Long full/compact defer1 same-history parity is running next. Independent pilot
must not start until both modes' long checks pass. Runtime repeat campaign,
formal QD3, candidate freeze, HD3, free generation and quality are still NOT_RUN.
No statistical/numerical promotion follows from these parity checks alone.

## Environment/provenance note

Initial small-run metadata recorded the collector's pre-launch affinity; the
actual model guard command explicitly applied taskset0–15. Long-run metadata
collection itself also applies taskset0–15, avoiding that ambiguity. All actual
deployed source/SO hashes remain in each server provenance JSON. No host library
or unrelated workload changes were made.
