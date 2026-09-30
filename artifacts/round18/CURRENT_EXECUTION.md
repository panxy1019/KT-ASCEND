# Round18 Current Execution

- Date: 2026-09-30 UTC
- Worktree: `/home/admin/Documents/ChatGPT/KT/kt-round18-native-preserving`
- Branch: `codex/round18-native-preserving`
- Parent commit: `6d1961c2caaad549c7609d13b5e3e9e37d8f9de6`
- Status: `ROUND18_COMPLETE_NO_PROMOTION`
- Default after Round18: unchanged Round17 S0

## Terminal disposition

Round18 is closed. Phase I and Phase II both reached evidence-backed negative
dispositions; no further PIECEWISE, device probe, full-model pool build or
request comparison is pending in this round.

### Phase I

`ROUND18_NATIVE_HYBRID_WORKS_NO_GRAPH_RECOVERY`

- Native Q32 partial passed in `native-partial-v004`.
- The Native-preserving CPU complement passed in `native-hybrid-v001`.
- The 48-layer eager smoke passed in `native-hybrid-full-smoke-v003`
  (remote snapshot alias `native-hybrid-full-v003`).
- PIECEWISE is locked after independent GDN and HyperConnection regions
  reproduced the frozen stack's dynamic scalar/range-to-shape ABI failure.
- Breakable is unsupported by the frozen stack.
- FULL replay does not re-enter the Python quant-method seam needed to create a
  fresh CPU work ID and result for every token and layer.
- No graph-safe candidate existed, so Phase I request ABBA was not run.

### Phase II

`ROUND18_EXPERT_POOL_GRAPH_COMPAT_BLOCKED`

- Fresh `native-route-trace-v016` produced 8,784 route events over W1/W2/W4,
  all 48 layers and exact requested output lengths. It is diagnostic route
  evidence, not a performance run.
- Q32 is the only tested capacity-safe pool under the frozen 55.5 GiB whole-card
  cap. Q64/Q96/Q128 exceed it under the same-class runtime estimate.
- A static W1-frequency Q32 plan does not supply all W2/W4 Decode experts.
  LRU/Belady reduce theoretical miss traffic, but require current/future route
  knowledge unavailable to the frozen FULL replay path.
- The bounded v018 device probe confirmed that the actual Physical-Q32 `w13`
  and `w2` parents are NZ29, then failed closed before mutation because the
  frozen stack has no qualified parent-preserving fixed-slot writer.
- Source review found a graph-external mutation seam and an independent
  per-expert tensor-list consumer, but no EP1 Host-backed owner, current-token
  device planner, or proven weight-then-map publication path.
- A bounded independent-list component probe is preserved as
  `OFFLINE_READY_DEVICE_NOT_RUN`. It was deliberately not run after the fresh
  route/capacity/graph economics gate rejected full implementation.
- W8A8 and preprocessed backing were not entered because the planner/graph gate
  had already failed and no supported blob-to-slot writer existed.

## Performance status

No Round18 candidate passed the correctness, capacity and graph-compatibility
gates required to run formal request performance. Consequently:

- `NONE -> GRAPH -> GRAPH -> NONE`: not run;
- `S0 -> BEST18 -> BEST18 -> S0`: not run;
- no Round18 speedup or slowdown is claimed;
- Round17 S0 remains the default and its historical performance is background
  only, not a fresh Round18 paired result.

## Resource terminal state

- Latest device block: `native-expert-pool-fixed-slot-v018`.
- Container exit/removal: PASS.
- Physical12 process after exit: none.
- Registered-file cache retirement: PASS after evidence/SHA archival.
- 179 registered read-only model/bank files checked; 342,753,280 resident bytes
  retired with `POSIX_FADV_DONTNEED`.
- Global `drop_caches`: not used.
- External tasks: not modified.
- Historical timers: remain paused.

## Reopen contract

Expert pool work should be reopened only when the same frozen or explicitly
requalified stack can provide both:

1. a device-side current-token miss planner without per-token Host route
   readback; and
2. a format-aware fixed-slot writer or parentless independent-slot owner that
   publishes a stable device map only after all expert projections are ready.

The offline independent-list probe is a bounded future entry point, not an
unfinished Round18 action.
