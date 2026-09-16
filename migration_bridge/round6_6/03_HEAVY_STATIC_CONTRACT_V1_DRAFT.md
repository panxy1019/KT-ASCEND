# Heavy Static Contract V1 — draft, execution not yet authorized by B0

This is an independent experiment identity, not an amendment of Round6.5's failed B_pair2.1875 contract. Draft written before new QD6/HD6 model outputs. The supplied scope refers to a prior V1 plan, but no such artifact was located in this checkout or the Downloads directory; this document makes the executable assumptions explicit rather than pretending a missing contract was already frozen.

## Exact semantics and operator gates

* Same frozen source weights, logical IDs, disjoint CPU/NPU ownership, physical layout, per-route weighting, shared/scaling ownership, framework transfer lifetime, finite output, same-path determinism, clean restart, CPU-not-hit exact control.
* Cross-path router IDs need not match after their hidden inputs differ. Router correctness means the correct router function/TopK on each path's actual operand, not forcing baseline routes into production.
* B0: actual CPU/GMM and canonical-reference decomposition, native-stage observer parity, production-size NPU layout control, real decode/prefill coverage. Local relative L2 safety stays0.01.
* Production default remains LLAMAFILE; no reference lane or NPU-for-CPU diagnostic in serving.

## Model-level study

QD6 and HD6:12 new prompts each,64 matched-history positions each; two examples per family (English facts, Chinese facts, numeric, structured, code, reasoning). Freeze both input/token sets together before QD6 execution. Exclude all old Q/H/root-cause/operator-only prompts from final HD6.
Use complete baseline histories; audit every forward including final lookahead and startup boundary. Logits, router-set Jaccard, CPU route count/weight mass, first routing divergence, and layerwise amplification are reported. QD6 describes the distribution; its maximum is not multiplied to manufacture a new universal bound.

Under this new four-layer design, model-distribution metrics are diagnostic rather than a replacement global max-pairwise gate. Finite values, operand/capture integrity, exact system semantics and local numerical safety remain hard gates. Large changes require causal review in the ADR; they cannot be hidden by top1 agreement or treated as independent evidence of quality. Candidate freezes this interpretation before HD6; HD6 cannot be used to rewrite it. QD6/HD6 are not population-wide guarantees.

## Independent paired task-quality screen

Proposed minimal-cost scope:32 GSM8K +256 CMMLU, selected deterministically from the frozen source pool, excluding prior fast-quality primary and dev IDs. Retain original prompt construction, token budgets512/32, parser, stop strings, and invalid/truncation-as-wrong treatment. All-NPU and four-R32 run on the same910B environment; do not import old-device scores.

Conservative screen criterion: no observed net accuracy loss on either endpoint; no increase in invalid count, no nonfinite/request errors, no catastrophic subset collapse. This does **not** silently inherit Deferred's2pp allowance. Report paired discordance, exact-marginal confidence intervals and paired bootstrap diagnostics. Passing the screen is explicitly not a population noninferiority claim; low reference accuracy or substantial truncation requires a scope-limited ADR, never 'quality is universally unchanged'. A failed screen is preserved; do not change the sample set after observing it.

## Sequencing / scope of qualification

B0 complete → freeze executable contract/corpora → expanded system/local audit → QD6 → candidate freeze → one-shot HD6 → free generation and paired quality → ADR.
Only four-R32 is initially qualified. All-MoE capacity, R16/R8, Static Frequency, Dynamic and Deferred are not inherited from it. Round6.5 historical status and old thresholds remain unchanged even if this new experiment succeeds.

Status: DRAFT_ONLY; QD6=NOT_RUN; HD6=NOT_RUN; quality=NOT_RUN.
