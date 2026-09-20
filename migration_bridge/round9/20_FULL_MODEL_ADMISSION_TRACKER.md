# Progression endpoint: full-model resource admission

User authorized continued monitoring, repair and development until this gate.
Target110.123.0.3 physical15. Thread heartbeat `round9` runs every30minutes;
this is periodic follow-up, not a continuous device-monitoring daemon. Refresh
telemetry before each run and monitor every owned NPU run separately. Original
`round7` remains PAUSED. Delegation uses gpt-5.6-sol / medium.

Do not leave unknown active workers between runs. Reuse exact source paths and
read CURRENT_EXECUTION plus git status before changes. Preserve .22 uncommitted
work and its saved archive; no training/global-mode changes, push, main or tags.

## Current admission dependencies

| Gate | Current evidence | Remaining |
|---|---|---|
| Source identity | Original full verification receipt + live144file sizes/config/index hashes PASS | Recheck on each run |
| Source reconstruction | Exact SGLang f02 pin recovered in independent .3 checkout | Reproducible runtime installation/import manifest |
| Source semantics | Tensor census and GR audited | QSA/GDN/PLE and KTEP integration |
| Runtime access | Shared physical15 mapping and bounded runtime PASS | Fresh per-run mapping/telemetry; no standing reservation |
| CPU C64 | Prior bounded BF16/F32 actual-weight probes PASS | Real activation and complete ownership/layout gates |
| GR | Isolated real-weight FP32/BF16 matrix PASS | Connect to model runner; final mixer coverage |
| QSA | NOT_RUN | Exact selector/indexer/causal KV reference and NPU tests |
| GDN | NOT_RUN | Chunk/decode/state reset/cancel/isolation tests |
| Synchronous PLE | NOT_RUN | Exact hash/history/row gathering/norm/gate/dilated convolution |
| KTEP ownership | Partial CPU skip/restore evidence only | CPU/NPU disjoint complete ownership and scaling exactly once |
| Layout/scaling | NOT_QUALIFIED | Packed axes and ND/NZ real-weight roundtrip |
| Full-model budget | NOT_QUALIFIED | Complete load/steady HBM envelope, host/disk/NUMA, bounded concurrent workload |

The frozen reference is Q64, BF16 storage/F32 CPU compute, C64=64workers+8aux,
text TP1, synchronous PLE, batch1/max-inflight1. Routed resident weights alone
are28800MiB. Account separately for other weights, KV/state, runtime workspace
and loading temporaries. Current free HBM cannot stand in for a training peak.

## Admission tooling

`tools/full_model_admission.py` is a pure evaluator, not a model launcher.
It layers the explicit full-model prerequisites over the existing device,
telemetry, HBM and physical-CPU checks. Input receipts must already be verified
by the caller; a syntactically valid hash is not independently verified content.
Never populate missing receipts from local test fixtures or hypothetical budgets.
No actual full-model PASS has been produced.

Test coverage: runtime-only evidence rejected; all required gates individually
missing rejected; stale telemetry, wrong profile, unbounded training peak,
insufficient RAM/disk, malformed HBM components and CPU topology rejected.
Tests first failed with missing implementation, then passed; malformed input
regression exposed TypeError and was fixed to return BLOCKED. General suite42.

## Next execution batch

1. Establish portable synchronous Qwen4 PLE hash/history and row-gather canonical
   from upstream4ccff141, including layer_id+1 mapping and128shard ordering.
2. Test against independent examples and actual checkpoint rows, then NPU
   normalization/gating/dilated convolution within the bounded probe envelope.
3. GDN and exact QSA independently, followed by runner integration and KTEP
   boundary gates; preserve all failures and freeze tolerance before measurement.
4. Only after all prerequisite receipts pass, construct a fresh real full-model
   budget snapshot. If training peak remains unbounded, record BLOCKED_RESOURCE
   and request the specific missing scheduling condition rather than guessing.

The endpoint is admission to begin full-model loading, not completed inference,
CMMLU, lowbit, C128, long-context or performance qualification. On reaching it,
report evidence and stop automatic advancement beyond this requested endpoint.

## Admission review corrections

Independent review (gpt-5.6-sol / medium) reproduced a false PASS with1MiB
self-declared host/disk requirements and unbound prerequisite receipts. Negative
tests reproduced the false admissions before correction. The evaluator now:

- Requires640–768GiB owned host budget,64GiB host reserve, and1.2TiB free working
  allocation under this initial conservative policy. These are admission
  reservations, not measured consumption. A lower budget needs a separately
  reviewed policy change supported by measured loading/steady-state evidence.
- Binds gate name, source revision, profile and runtime implementation digest.
- Requires a separately identified budget envelope binding host/disk/HBM values
  to batch1/max-inflight1 and explicit context/output limits. Test fixture values
  are synthetic and must never become operational receipts.
- Requires an unexpired device-specific concurrent-workload bound matching the
  numeric training peak, with an enforced reservation or independently verified
  workload-envelope basis. A current idle snapshot or a bare boolean cannot pass.

The pure evaluator trusts independently verified receipt metadata from its
caller; it does not validate files or grant a reservation. An operational
launcher must load/hash and validate actual receipts before supplying this input.
No production full-model launcher or admission PASS has been created.
All42 tooling tests pass after the corrective pass. No repeat NPU test was
needed: this batch changes only host-side admission policy, not device math.
