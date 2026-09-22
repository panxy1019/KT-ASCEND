# Single-layer compact ratio qualification — PASS

Update2026-09-16~12:50UTC:R32/R16/R8 compact gates complete. R16/R8 each
have28exact determinism/prefix/restart requests and512matched-history positions;
259stable exact,253ambiguity membership,zero overflow,maxpairwise0.6875.
R8 additionally exercised the new runtime physical-weight hook:26layers checked,
Layer17resident8,other25layers resident64. Earlier R32/R16captures did not run
that hook and must not be retroactively described as physical-hook verified.
Actual-serving-token reanalysis of R32/R16 agrees with their earlier summaries.
All profiles have captured-real-input expert repeats and controlled nohit tests.
This is compact single-layer evidence,not final all-MoE qualification.

Target910B3 NPU0,32workers/NUMA0,production overlap,Deferred0. No production patch.

## R32 observed results

- Four frozen S6 prompts,3×64 repeats plus8/16/32 prefixes:24requests exact.
- Independent restart4×64:exact vs first process. Total28requests PASS.
- Matched-history Q2/H2 compact512positions:stable259/259 exact,
  ambiguous253/253 membership,zero overflow,max pairwise0.625 vs frozen2.1875.
- Whole-forward capture520=512targets+8lookahead,integrity PASS.
- Synthetic hidden/realE1 weights:T1/T4,10repeats deterministic;T4 relativeL2
  0.00372754003 (<.01);controlled accelerator-owned route CPU output exactzero.
- Natural diagnostic4×8 outputs exactly match uninstrumented prefixes.
  36forward records include4lookaheads. Routing/FP32 merge controls PASS.
  Real token rows:prefill46,decode28,lookahead4 (padded kernel log lengths are
  not actual prompt lengths). One naturally CPU-not-hit row verified exact.

These are compact single-layer results, not all-MoE qualification or final
performance. Physical sparse weight shape evidence is still to be consolidated.

## Captured control harness correction

r65-scale-matrix-v1 stopped before escalation withCAPTURED_HIDDEN_TOO_SHORT.
The new tool incorrectly required the first short prompt alone to have32rows.
This is an instrument assumption failure, not a provider/model failure.
v2 gathers first32actual chronological captured rows across calls, stores source
hashes, applies the same controlled NPU-owned route, and never pads/duplicates.
T1/T4 expert comparison still uses first rows of the unchanged first capture.
Frozen masks/corpora/B_pair/production arithmetic are unchanged.

The earlier pending R32captured-control state was closed in v2;R16/R8 then
completed as described at the top. Later four-layer numerical failure does not
retroactively invalidate these compact single-layer observations.
