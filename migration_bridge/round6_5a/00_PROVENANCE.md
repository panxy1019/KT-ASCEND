# Round 6.5A — frozen diagnostic scope

Base: `b2e9d8721911402e94cf3b103f3e0d3800fd81be`.
Branch: `codex/round6-5a-static-numerical-rootcause`.

Round 6.5 remains `BLOCKED_HEAVY_OFFLOAD_RUNTIME` / `STATIC_NUMERICAL_SCALE_BLOCKER`.
This investigation does not promote any profile or change the acceptance contract.
The inherited 825-artifact manifest passed the blocked-freeze validator at entry.
The pre-existing dirty `third_party/llama.cpp` worktree is preserved and excluded.

## Historical fixture

- four-R32: layers 1, 9, 17, 26; 32 resident experts per selected layer.
- `h2_en_01`, token index 54; serving top1 82 on both paths.
- Frozen bound 2.1875, candidate K=32; observed distortion 9.59375.
- 27 candidate-pair overflows; top1 agreement is not an override.
- Original artifacts remain under `../round6_5/evidence/` without modification.
- Runtime parent e3084d309d6f39b8a9f7185d36ac27e4b77526d7 and child
  eaa58ccea4ee372b8b207dd16f398eafe3da37ff include recorded overlays.
  Their imported-file and binary hashes, not this diagnostic branch HEAD,
  define the historical executable.

## Execution checkpoints

1. Freeze fixture and revalidate actual runtime provenance.
2. Compare the real grouped-MoE R64/R32 resident layouts on identical operands.
3. Compare production against a canonical mixed-expert oracle.
4. Bisect cumulative layers; locate complete decoder-state divergence.
5. Replay router and counterfactual routes; run diagnostic NPU-for-CPU oracle.
6. Audit arithmetic, scaling, casts and reduction topology; issue root-cause ADR.

No production patch before an invariant defect is demonstrated. Any patch must
have a negative control and pass the original four-R32 gate before resuming 6.5.
Otherwise the final classification is INHERENT (only with all required causal
evidence) or UNRESOLVED, with Round 6.5 still BLOCKED.

Deferred/Dynamic remain OFF; 32 workers remain fixed. No R16/R8 promotion,
all-MoE sweep, quality run, threshold fitting or core expansion in this diagnostic phase.

## Interpretation guard

An exact current-token Layer1 MLP output alone does not prove exact full decoder
state. Attention/KV history and the separately carried residual must also be
compared. This is an investigation hypothesis, not a root-cause conclusion.
