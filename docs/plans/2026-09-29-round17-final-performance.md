# Round17 final performance execution plan

Round17 is the final KT-ASCEND performance optimization round.  The frozen
S0 configuration remains the control.  This round changes only work that can
shorten the complete request critical path.

## End state

The round ends in one of two evidence-backed states:

- `ROUND17_FINAL_PERFORMANCE_GAIN_QUALIFIED`: at least one QSA-parent or exact
  NPU-segment candidate passes component, quality, resource, and fresh request
  ABBA gates.
- `ROUND17_COMPLETE_NO_NEW_PROMOTION`: both bounded directions have an
  implementation or evidence-backed disposition and no request candidate
  passes the frozen gate.

`ROUND17_PARTIAL_WITH_GAPS` is reserved for an external resource, numerical,
state, or interface condition that prevents the required real-request
decision.

## Frozen control

The parent is commit `6df066bbca6f94e627a549f2cc5cfb51b0c5c8cb` and
the S0 runtime recorded in `artifacts/kt-perf-root-cause/BASELINE_MANIFEST.json`:
Qwen3.8 revision `de4b8e4d43b917e7706784d8bb445c9af86a3540`,
C128/MXFP4, Uniform Q32 BF16/NZ29, chunk32, promoted GDN, QSA Vectorized
Prefill plus DIRECT_BF16_ROW Decode, PLE ASYNC, H129, TP1, concurrency1.

CPU Expert, TaskQueue synchronization, Dynamic, deferred, placement capacity,
Layerwise transport, vLLM migration, W8A8, MTP, concurrency, and long context
remain outside scope.

## Execution checkpoints

1. **P0 identity and fresh control.** Restore exact S0 source/runtime identity,
   run 2x10 second admission before each device start, and capture fresh
   profiler-off W1/W2/W4 controls used by any request comparison.
2. **P1 critical-path refinement.** Run two fresh diagnostic blocks.  Split all
   12 QSA parents into input projection, indexer, KV preparation, sparse GQA,
   output/state work, and split decoder glue around attention, expert, and
   HyperConnection boundaries.  Host and device clocks remain separate.
3. **A QSA parent.** Select at most two causes from P1.  A candidate reaches a
   request run only when the complete parent improves by at least 20% in both
   component blocks and preserves causal/GQA/state/fallback/lifecycle.
4. **B exact NPU segment.** Select at most two segment layouts.  CPU Expert
   stays outside capture and each Decode token receives a fresh CPU work ID.
   A candidate reaches requests only when complete-layer median improves by at
   least 15%, launch count falls, and no wait moves outside the measured parent.
5. **Fresh request decision.** A uses an 8% W2 TPOT and throughput gate; B uses
   10%.  Both require two paired blocks, W1/W4 protection, complete output,
   quality/state/cancel/reuse, Host peak, owned/whole-card HBM, and profiler off.
6. **Combination and freeze.** Combine only independently promoted changes,
   remeasure rather than composing percentages, then write final status,
   benchmark, source/build identity, raw evidence hashes, and Chinese report.

## Rulings

- The user-supplied Round17 plan is the approved design and authoritative
  acceptance contract.
- Diagnostic observer changes are default off and may add timing events only;
  they may not change model math, synchronization, routing, precision, or state.
- A direction without a component candidate that reaches its frozen gate is
  closed with a measured disposition; request budget is not spent on it.
- Repeating an unchanged failure is prohibited.  A retry must have a new root
  cause, source identity, or evidence boundary.
