# Dynamic DEV threshold256: full-model diagnostic qualification

2026-09-18 01:00 UTC checkpoint. Evidence: A3_VERIFIED.
This closes one DEV migration probe, not Round7B or the scheduler product gate.

## Frozen execution identity

- Owned container: `kt-r67-c64-development-v1`, physicalNPU2/logical0.
- TP1, batch1, C64=two32-worker subpools on CPUs128–191/NUMA4,5.
- Uniform R8 initial profile SHA:
  `1108791921bae5f6f59794d7295c7cb05d903bd8615ed8f97fc72527e231b05f`.
- DEV corpus SHA:
  `b79c132b095a0efd35c822f1b1124e57b666b1c5ffdf65f4933e7bcad02c694f`.
- Runtime-v7 and tools-v21, with unchanged model/GGUF/quality identities.
- Threshold256 is a pre-registered DEV candidate, **not the selected threshold**.
- Protocol sequence: dev_000, dev_067, dev_067, dev_066, dev_000.

## Complete probe result

`backing-requalification-v6/dynamic-dev-probe-256/summary.json`: PASS.
Summary SHA:
`1f1031df24da8b206c42b8ba6f4f7d4bf36d208fd9ba030ae74994519cd42536`.

| Gate | Result |
|---|---|
| Short input no-trigger and unchanged initial resident state | PASS |
| Real full-device/layerwise prefill trigger | PASS |
| Exact migrated weights and independent checkpoint sampling | PASS |
| Weight/mapping/mask/wrapper generation consistency | PASS |
| Repeated greedy trajectory and selected expert determinism | PASS |
| Workload A→B adaptation | PASS, resident identities changed in26/26 layers |
| No-trigger request after update preserves state | PASS |
| Independent restart restores initial static state | PASS |
| Restart-trigger trajectory and expert selection exact | PASS |
| Observer-OFF triggered trajectory parity | PASS |

Process A completed5 requests and78 audited updates. Process B completed2
requests and26 audited updates. Observer-OFF completed4 requests; lightweight
telemetry correlates78 updates to three actual triggered request windows and
reports one no-trigger request. All three processes have COMPLETE lifecycle and
`reaping.complete=true`. Intended graceful shutdown/reaping is not a crash.

The A→B experiment changes only the workload, not expert selection using quality
answers. Every update retains8 resident slots per layer. Recorded copy payload
for78 updates is10,796,138,496 bytes (624 copied expert slots); this includes
recopying resident identities and is not equal to unique changed-weight bytes.

## Evidence preservation

Lossless complete raw probe, configurations, protocol, summaries, generation/
consumption and routing records, checkpoint boundary audit, server logs,
lightweight telemetry and lifecycle/reaping:
`../raw/dynamic-dev256-runtime-v7-complete-v1.tar.gz`.
SHA:
`ca6fbd3f16d8496b7706f6fcd3bde406ee28281a06ce933f0e81b9abd789497f`.
Complete SCP exit0, matching A3/local hash and full tar validation exit0.
The previous runtime-v6 Layer2 failure remains historical FAIL; it is not rewritten.
Storage-offset warnings remain in the preserved logs. Completed diagnostic
identity checks and observer parity pass; warnings alone are not proof that
every conceivable shape or future concurrent execution is safe.

## Limits and next gates

Diagnostic weight readbacks add synchronization; update host-call duration is
not formal migration latency. Observer-OFF parity uses lightweight telemetry,
not the formal hook-free timing configuration. Only serial TP1/batch1 is covered;
no production concurrency, Graph, Deferred or multi-NPU qualification is inferred.

At checkpoint the healthy driver is running threshold512, followed by1024.
The owned watcher may select a DEV candidate only after all three complete PASS
probes. Commit and verify that candidate before HD, quality or performance.
Dynamic default remains unselected; Deferred remains OFF; Round7 is not COMPLETE.
