# Live progression: admission, then full-model testing

Latest user instruction pauses the Round9 heartbeat and requests live progression
through full-model testing. Both round9 and round7 schedules remain PAUSED.
Target110.123.0.3 physical15; delegated agents gpt-5.6-sol / medium.
Admission remains mandatory before full-model loading.

Do not leave unknown active workers between runs. Reuse exact source paths and
read CURRENT_EXECUTION plus git status before changes. Preserve .22 uncommitted
work and its saved archive; no training/global-mode changes, push, main or tags.

## Current admission dependencies

| Gate | Current evidence | Remaining |
|---|---|---|
| Source identity | Original full verification receipt + live144file sizes/config/index hashes PASS | Recheck on each run |
| Source reconstruction | Exact SGLang f02 pin recovered in independent .3 checkout | Reproducible runtime installation/import manifest |
| Source semantics | Tensor census; GR/QSA/GDN/PLE equations and original-weight module gates PASS | Model boundary and runner integration |
| Runtime access | Shared physical15 mapping and bounded runtime PASS | Fresh per-run mapping/telemetry; no standing reservation |
| CPU C64 | Real BF16 bank/native wrapper C64, 9 cases ×3 PASS; all48 banks exact verified | Full-model activation and host budget |
| GR | Isolated real-weight FP32/BF16 matrix PASS | Connect to model runner; final mixer coverage |
| QSA | CPU6 PASS; original-weight NPU30 PASS; actual pool boundary/lifecycle tests PASS | Actual adapter/chunk/decode and runner integration |
| GDN | CPU5 PASS; real-weight NPU18 local PASS | Lifecycle and runner integration |
| Synchronous PLE | Original hash/256 boundary rows and NPU6 local PASS | Packed requests, cancellation/reuse and runner integration |
| KTEP ownership | Actual KTEP Q64/C64 merge 9 cases ×3 PASS; disjoint zero ownership and FP32 partial sum verified | Shared expert and router/scaling at model boundary |
| Layout/scaling | All48 BF16 banks 73728 projection identities; layer0 all64 resident ND/NZ roundtrips PASS | Full model packed loads and scaling |
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

1. Finish real SGLang adapter whole/chunk/token-decode and request reuse tests
   for GDN, PLE and QSA indexer. Preserve every failed attempt and exact repair.
2. Qualify shared expert/gate, routed scaling and final GR at model boundaries;
   close real-weight loading and runner binding. Meta mapping is not real load.
3. Build source/profile/runtime-bound prerequisite receipts and fresh resource
   accounting. Do not treat idle physical15 as a guaranteed training envelope.
4. Proceed to full-model tests only after actual admission. An unresolved training
   envelope requires a concrete scheduling condition; never invent a reservation.

The latest endpoint is full-model testing after admission. CMMLU, lowbit, C128,
long-context and performance remain separately gated; no earlier-model scores
or device qualifications are inherited.

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
