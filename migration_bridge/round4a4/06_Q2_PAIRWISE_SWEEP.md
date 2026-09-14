# Q2 Pairwise Teacher-Forced Sweep

Evidence state: `A3_VERIFIED`

Q2 contains 18 prompts and 1,152 matched-history positions. At every position,
All-NPU and Hybrid consumed the same frozen All-NPU token prefix. The candidate
universe was the union of the baseline top-32 and Hybrid top-32 tokens; no
Hybrid-emergent top-32 token was discarded.

The sweep produced 36,104 baseline-top1-to-candidate pairs. All captured logits
and derived margins were finite. Every sampled position exercised at least two
P1 CPU-owned routes, so the separately captured 32-position CPU-not-hit control
is the exact negative control.

Artifacts:

- `evidence/q2-allnpu-history.json`
- `evidence/q2-union.json`
- `evidence/q2-pairwise-metrics.json`

Corpus SHA256: `551fc4fdf75053eda18d511ab2b479907576bd31989460281ef2424689dd3e07`.

