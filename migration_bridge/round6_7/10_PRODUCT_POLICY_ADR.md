# ADR: new serial all-MoE product screen

Status: **PROTOCOL ACCEPTED BY USER; RESULTS PENDING**. 2026-09-17.

The preceding strict numerical contract did not extrapolate from low-offload
P1/P2/P3 to four-R32. Root-cause investigation classified cross-backend
numerical amplification and router discontinuities, not an observed runtime
invariant defect. Round6.6/B then tested independent numerical and quality
protocols; their rejected conclusions remain unchanged.

The user now explicitly chooses a **product policy**, CMMLU-only with maximum
5pp observed loss, to screen capacity-oriented all-MoE ratios. This is a new
decision axis and experiment identity. It neither repairs the old B_pair gate
nor turns historical GSM8K/invalid-output failures into PASS.

## Exact and local safeguards retained

Checkpoint value identity, global IDs, disjoint ownership, physical mapping,
route-weight/scaling/shared-expert exactly-once semantics, framework-managed
transfers, same-path greedy/prefix/restart determinism, finite results. Existing
production semantics are unchanged. Sampled same-operand CPU vs Ascend
grouped-MoE relL2 remains <= .01. Quality tolerance cannot cover a kernel error.

## New bounded product screen

Fresh R64 and all Hybrid profiles use exact same CMMLU256 prompts/input tokens,
5-shot letter-likelihood protocol, tokenizer, model and runtime. Product gate
is observed delta >= -5pp and no invalid/failed requests. A256 sample has coarse
resolution and limited statistical power: report pairwise bootstrap uncertainty,
do not claim population-level non-inferiority or task-general quality.

No raw cross-backend token/router equality or low-offload B_pair gate is used
to declare product PASS. Same-path nondeterminism, NaN, mapping/ownership
defect, local expert gate failure still fail SYSTEM and stop that profile.

## Capacity/performance decision

Four serial profiles, frozen nested uniform masks, all26MoElayers. Capacity
first; no undocumented speed floor. Only product-accepted candidates receive
formal serial observer-OFF performance. One container is maintained, no
multi-NPU qualification. Topology is fixed but CPU NUMA0/NPU NUMA6 is nonlocal.

Final selection must report actual HBM versus throughput and quality, and name
a Static Frequency **entry candidate**, not a Dynamic-ready configuration.
No Static Frequency or migration is fit using CMMLU correctness output.

If none passes, close with NO_QUALIFIED_HEAVY_BASELINE. Do not change masks,
items, tolerance, workers or decoding after seeing results. Independent future
low-bit/frequency/dynamic studies need new experiment identities.
