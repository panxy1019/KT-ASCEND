# Round10 Hybrid, Kunpeng, Dynamic, and Qwen narrow qualification

Date: 2026-09-26

This is the executable local form of the user-approved Round10 route. It starts
from the immutable v90 Hybrid129 snapshot and advances through T0–T6. It does
not reopen the rejected Round9 Layerwise/NZ update candidates.

## Fixed stage order

1. **T0 — restore and freeze H0.** Verify v90 source, model, tokenizer, bank,
   resident map, providers, process identity, selection logic, resource gates,
   and a new append-only budget.
2. **T1 — Hybrid policy and quality.** Qualify actual input lengths up to 128
   tokens, natural outputs up to 64 tokens, serial request execution, policy
   edges, multi-step decode, cancellation, cleanup, and same-process reuse.
3. **T2 — Hybrid cost attribution.** Capture replayable real fixtures and
   separate request, CPU submit/worker/wait, resident NPU, merge, stateful-model,
   and exposed critical-path costs. Diagnostic synchronization must not be used
   as unprofiled request performance.
4. **T3 — Kunpeng Expert core.** Reach at least one native worker path beneath
   the Python submit/sync seam, verify it numerically, and evaluate at most two
   single-variable candidates. Promote at most one. No gain leaves the existing
   CPU backend enabled and does not block T4.
5. **T4 — Dynamic.** Exercise a real Hybrid request lifecycle, prove placement
   transaction/epoch/cancel/reuse behavior and precision, then measure net
   benefit. A correct candidate without net benefit remains disabled.
6. **T5 — Qwen B path.** Independently validate GDN, QSA, and PLE providers,
   references, pool state, state continuity, cancellation, and reuse. Evaluate
   at most one B-module performance candidate and only when evidence identifies
   a hotspot.
7. **T6 — integration.** Build HFINAL from individually qualified changes,
   rerun narrow quality/cancel/reuse and unprofiled measurements, freeze hashes,
   retire owned resources, and issue the truthful terminal status.

## Stage rules

- Correctness and evidence pass advances to the next stage.
- Correctness without benefit disables the candidate and advances on the
  parent baseline.
- A candidate failure is preserved, diagnosed, minimally repaired, and tested
  under a new identity. The same root cause receives at most two revisions.
- A reproducible baseline error blocks performance promotion until repaired.
  Independent offline and component work may continue.
- Resource denial triggers cleanup, bounded backoff, and offline work. External
  processes and containers are never modified.
- Missing independent reference or real trigger blocks that qualification item;
  it is not converted into a pass.
- Exhausted stage budget records a gap and does not borrow reserved work from
  another required stage.

## Immutable H0 dimensions

- Qwen3.8-Flash-Next revision
  `de4b8e4d43b917e7706784d8bb445c9af86a3540` and the v90 tokenizer/template.
- C128: four NUMA pools with 32 workers each; actual CPU affinity comes from
  the v90 receipt. The wider cpuset is not a claim of 160 workers.
- Read-only MXFP4 CPU expert bank, Q32 resident BF16/NZ29 map, TP1, serial
  execution, scheduler chunk32, prefix/radix/graph/MTP/overlap disabled.
- Coupled threshold129 profiles, PLE ASYNC, and Dynamic disabled as actually
  recorded by v90.
- Original shadow32 Layerwise implementation and synchronization remain
  available for the boundary route. R2, R3, transpose-view, and shadow320 stay
  disabled.

Each candidate records its parent manifest and a single behavior change. Speed
ratios from independently tested candidates are never multiplied to predict
HFINAL.

## T1 device schedule

T1 may use at most two model starts and 24 generation requests. Each start has
a frozen request queue before launch. Process A covers five deterministic
canaries, H64 warmup, two H64 measurements, multi-step decode, cancellation,
and reuse. Process B covers 63/64/65/127/128 edges, multi-step decode, H64
warmup and two measurements, cancellation, and reuse. Two requests remain for
one optional 129/130 Layerwise boundary probe or one predefined quality gap.

Every device start requires a new two-round, 60-second resource admission.
Every run must reach guard exit acknowledgement and owned-container removal
before the next start. Automatic retry is zero; a controller defect uses a new
run identity and budget entry.

## Terminal outcomes

- `ROUND10_AB_NARROW_QUALIFIED`: T1–T6 required correctness items and HFINAL
  integration pass. This does not imply broad product/deployment qualification.
- `ROUND10_PARTIAL_WITH_GAPS`: one or more required Dynamic or B-path contracts
  remain unproved within their bounded stage budgets.
- A safety, identity, or cleanup breach stops new device submission and is
  reported as the actual terminal state.

