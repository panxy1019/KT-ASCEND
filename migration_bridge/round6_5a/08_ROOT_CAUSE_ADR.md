# ADR — Round6.5A static numerical scale root cause

Decision: **INHERENT_CROSS_BACKEND_NUMERICAL_AMPLIFICATION**.

Scope: the frozen `four-R32 / h2_en_01 / token54` trajectory and the bounded
operator/layout/prefill checks documented here. This is an empirical causal
classification, not a proof that every model input and kernel shape is bug-free.

## Evidence chain

1. Historical failure/observer/repeat/transfer evidence remains immutable.
2. Deployed source/extension provenance matches the inherited manifest.
3. Four layers ×eight resident experts: R64/R32 same-input grouped-MoE output
   exact, with ten identical repeats per layout.
4. Canonical mixed reconstruction: all four failing-token routed outputs and
   all four real prefill routed outputs match production after the declared
   final cast. Complete CPU replay partials match their captures.
5. New per-expert CPU-versus-GMM maximum relative-L2 ≈0.000517213; prefill
   maximum ≈0.000434630. Both remain below the unchanged 0.01 local gate.
6. Same deployed router operator exactly reproduces both logits/Top-K sets.
   Layer17 E62/E29 score ordering reverses; E29 rises from rank8 to rank6.
7. Baseline-route intervention reduces the Layer17 jump and overall logit L2,
   while worsening worst pairwise distortion. Neither result is hidden.
8. NPU-for-CPU substitution preserves R32 ownership/control flow and restores
   64/64 complete logit vectors to exact all-NPU output, with zero overflow.

The machine-readable decision checks these evidence predicates before writing
the classification. Nine model/operator process guards completed and reaped
their owned descendants.

## Important correction to the initial mechanism hypothesis

C1 (only Layer1 offloaded at R32) already reproduces token54's failure and the
Layer17 router-set switch. Multiple CPU-enabled layers are not necessary.
The previous single-layer passing profile used Layer17, so it did not establish
that arbitrary single-layer R32 placement is safe under the old envelope.

The results support small early backend perturbations propagating through
subsequent model state and routing discontinuities. C1's prefill output already
differs slightly. At token54, Layer2 attention differs despite an exact current
input, consistent with different historical attention state. Direct KV-buffer
bytes were not captured; the precise KV propagation is not claimed proven.

## Implementation branches

- No resident-layout defect was demonstrated in the tested layouts/experts.
- No incorrect merge/cast invariant was demonstrated in the reconstructed
  decode/prefill cases.
- The inspected routed-scaling and shared composition follow the retained
  contract. Model scaling factor is 1.0, so multiplicity is established by code
  inspection rather than an output-sensitivity test.
- The run uses one subpool, not the upstream two-subpool reduction case.
- The exact arithmetic instruction/reduction difference is not established.

Accordingly, **no production patch is justified by the present evidence**.
Forcing routes, executing CPU-owned experts on NPU, or selecting a cast after
seeing this failure would change the experiment rather than fix an identified
invariant defect. Diagnostic hooks must not enter a production configuration.

## Historical acceptance remains unchanged

```
ROUND6_5_HEAVY_OFFLOAD_ALIGNMENT = BLOCKED_HEAVY_OFFLOAD_RUNTIME
BLOCKER_SUBCLASS = STATIC_NUMERICAL_SCALE_BLOCKER
FIRST_BLOCKED_PROFILE = four-R32
B_pair = 2.1875
K = 32
RESIDENT_RATIO_PARETO = NOT_COMPLETE
ROUND7_BASELINE = NOT_FROZEN
ROUND7 = NOT_STARTED
DEFAULT_DEFERRED = OFF
DEFERRED_HEAVY_PROFILE_PROBE = NOT_RUN_PREDECESSOR_FAILURE
```

There was no real implementation fix, so the original four-R32 qualification
was not rerun as a purported post-fix pass. Its failure remains authoritative.
R16/R8 promotion, all-MoE, quality/performance promotion and Dynamic were not run.

## Next independent study

If development continues, use a new Round6.6/6.5R experiment identity for heavy
static numerical qualification. Preserve exact system semantics, same-path
determinism, clean restart, CPU-not-hit and local operator safety. Pre-register
new derivation/held-out corpora, router-aware diagnostics and paired downstream
quality criteria. Exclude old qualification/held-out and root-cause prompts
from the new final holdout. Freeze before looking at held-out outputs.

Do not derive a replacement bound from 9.59375, 16.65625 or any present failure
maximum. Do not automatically reuse Deferred's approximate-execution quality
tolerance for static heavy offload. That study was not executed in Round6.5A.
