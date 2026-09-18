# Runtime-v7 closed predecessors and live Dynamic checkpoint

Checkpoint: 2026-09-18 00:44 UTC. Evidence state: A3_VERIFIED for
completed predecessors; Dynamic full-model qualification remains pending.

## Closed, freshly executed predecessors

The fresh `backing-requalification-v6` driver uses runtime-v7 and tools-v21.
No previous PASS summaries were copied into this run.

- Operator cases: 100/100 PASS; maximum relative-L2
  0.0017029279396063187, below the unchanged 0.01 local gate.
- Uniform: 32 compact requests and four independent restart requests PASS.
- Frequency: 32 compact requests and four independent restart requests PASS.
- Both profiles retain all four captured reference trajectories exactly versus
  runtime-v6. Repeat, prefix, finite and observer-parity checks PASS.
- Fresh direct All-NPU R64: 174/256, invalid=0. Resolved command matches the
  historical direct-NPU protocol; the reference is not silently replaced by KT64.

Lossless closed-predecessor archive:
`../raw/runtime-v7-closed-predecessors-v1.tar.gz`.
SHA256: `c1aee04bddc48c51969a45908b1c83462ba936b15ff02fd3e2a38c1f1c491845`.
SCP completed with exit0; local and A3 hashes match; complete tar listing passes.
Archive includes compact/restart raw results, process lifecycle/reaping records,
fresh R64 raw256 results, summaries, resolved configuration, runtime provenance
and operator summary. Large operator arrays remain on A3; live Dynamic files
were deliberately excluded to avoid archiving an incomplete stage as closed.

## Live threshold256 DEV probe: not yet PASS

At 00:43:52 UTC, process A completed five requests and 78 audited updates
(three triggers across all26 MoE layers). Process B completed its two restart
requests and 26 audited updates. All104 observed updates have exact weight and
checkpoint identities and equal weight/mapping/mask/wrapper generations.

The original Layer2 failure no longer occurs in these diagnostic requests.
The driver has advanced to observer-OFF parity, whose short initial request
completed and first triggered prefill is healthy. Its trajectory comparison and
remaining requests must complete before `dynamic_probe.py` writes a PASS summary.
A storage-offset warning is preserved in the logs; passing diagnostic readbacks
alone do not establish observer-OFF parity or formal performance.

Runtime binding is unchanged from `RESIDENT_TO_RAW_LAYOUT_REPAIR.md`:
wrapper86f0f168..., resident helperac2ac86a..., CPU extension46182176....
No model, GGUF, mask, threshold candidates, C64 definition or quality gate changed.

## Ordered continuation

Keep the healthy probe intact. The driver proceeds to thresholds512 and1024
only after256 passes. The owned watcher may start DEV-only threshold selection
only after all three complete/reaped PASS probes. Any eventual candidate requires
verified local Git commit contents and matching acknowledgment before HD, CMMLU
or performance consumers. No threshold is selected or held-out stage launched at
this checkpoint. Deferred remains OFF and unevaluated in Round7C.
