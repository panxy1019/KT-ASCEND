# Round 5 — Aligned P1 Requalification

Evidence state: `A3_VERIFIED_READY`

## Frozen profile

```text
model: deepseek-ai/DeepSeek-V2-Lite
revision: 604d5664dddd88a0433dbae533b7fe9472482de0
TP: 1
device: NPU0 only
CPU layer: 17
CPU experts: {6, 8, 25, 36}
physical NPU expert count: 60
CPU backend: LLAMAFILE / CPUInfer, F32 GGUF
NPU backend: Ascend BF16/unquantized
Graph/Deferred/Dynamic/MTP/Speculative: OFF
execution: production overlap
```

The aligned parent under test was `be5b48621` with SGLang child
`412219b4b05a888808f0b88d2963a8b6164a0fda`.  The parent commits after the
last runtime wheel build contain tests only; no unbuilt production source was
used by this qualification.

## System semantics: C0-C4

The real Layer 17 weights and the frozen P1 expert ownership were exercised
through the production `KTEPWrapperMethod` path.  Each routed result was also
constructed explicitly as `sum(CPU partials) + sum(NPU partials)`.

| Case | CPU-owned routes | KTEP vs explicit max abs | Relative L2 |
|---|---:|---:|---:|
| C0 NPU-only | 0 | 0 | 0 |
| C1 one CPU route | 1 | 0 | 0 |
| C2 two CPU routes | 2 | 0 | 0 |
| C3 CPU routes in different top-k positions | 2 | 0 | 0 |
| C4 four CPU routes / multi-token fixture | 4 | 0 | 0 |

Additional exact invariants:

- logical-to-physical mapping dtype is `torch.int32`;
- physical NPU expert count is 60;
- CPU and NPU ownership is disjoint;
- routing weights are consumed once by their owning provider;
- routed scaling and shared expert are each applied once outside the routed
  merge;
- C0 CPU-not-hit output equals the all-NPU reference exactly;
- the shared-plus-scaling final output equals the explicit reference exactly.

## Same-path determinism

The aligned serving profile completed:

```text
9 prompts x 10 repeats x 64 greedy tokens = 90 full requests
9 prompts x {8,16,32} prefix requests      = 27 prefix requests
```

Every prompt produced one unique 64-token trajectory.  Every 8/16/32-token
request was an exact prefix of its corresponding 64-token trajectory.  All
5,760 recorded generation logprobs were finite.

An independent serving-process restart then reproduced all 9 frozen
64-token trajectories exactly, with finite logprobs.

## Frozen pairwise numerical contract

The Round 4A contract was not refitted:

```text
B_pair = 2.1875
candidate K = 32
contract SHA256 = 223e738436659d389d913656952a368b6d32ccaceeef195abc2e6589c651d717
```

Fresh aligned logits were captured for the complete Q2 (1,152 positions) and
H2 (1,024 positions) corpora.  The historical full-logit baselines were never
retained for every position, so the actual frozen 256-position Q2 and
256-position H2 baseline subsets preserved by Round 4A were used for the
cross-architecture comparison.  No baseline was reconstructed or refitted.

| Corpus | Positions | Candidate pairs | Max distortion | Stable exact | Ambiguous membership | Overflow | Result |
|---|---:|---:|---:|---:|---:|---:|---|
| Q2 frozen subset | 256 | 8,052 | 1.75 | 137/137 | 119/119 | 0 | `Q2_QUALIFIED` |
| H2 frozen subset | 256 | 8,040 | 0.5625 | 155/155 | 101/101 | 0 | `HELDOUT_VERIFIED` |

All compared outputs were finite.  Ambiguity-set cardinality remained
diagnostic only, as required by the frozen contract.

## Downstream quality

The frozen quality manifest was used unchanged:

```text
c7bcb09c72f4a7213d0ecdb080f8e88e983ab1a491d4304f448bce28d8f7e1ce
```

| Benchmark | Frozen all-NPU | Aligned P1 | Invalid | Discordant correctness pairs | 20,000-pair bootstrap delta 95% CI |
|---|---:|---:|---:|---:|---:|
| C-Eval | 27/128 | 27/128 | 0 | 0 | [0, 0] |
| GSM8K MC | 31/128 | 31/128 | 0 | 0 | [0, 0] |

Result: `QUALITY_VERIFIED`; no statistically meaningful regression was
observed.

## Lifecycle

The production-overlap C0-C4 fixture ran for 1,000 forwards and produced one
unique routed-output hash:

```text
878e638f32c3311f816f3dc01f6e83dcd822c4f332083a591675d3ccaaafaba2
```

There was no crash, deadlock, traceback, or non-finite output.

## Evidence integrity

| Artifact | SHA256 |
|---|---|
| `a3_aligned_p1_repeat64.json` | `bc23b340a0ebce2a39ad1ce51a1933acf703dd296092ae3081721d6e7d8c45cb` |
| `a3_aligned_p1_same_path_summary.json` | `9fe718e544c180f2f065d6a02a8a0391815166aa800b4f70d23a8c5da358abce` |
| `a3_aligned_p1_finite_summary.json` | `102c9471dda11dfdefb2b894ac9950ae942c71f4559685eeef5e5121237e2cd8` |
| `a3_aligned_p1_clean_restart.json` | `09534d5805f51849933b40d52974c4a0a79e892d7b191e6b7cbf51434d996a66` |
| `a3_aligned_p1_clean_restart_summary.json` | `11c46dc3c6eefb907c74e32580fa09bee896a9880a823e82b2cc956963851c72` |
| `a3_aligned_p1_q2_contract.json` | `77ca53e1368a07f96bec22ad0394d23b0a4b278d5b10554aca5362104ddc9625` |
| `a3_aligned_p1_h2_contract.json` | `176b10d64e3d19016f9ae33c3658dccada609829b4d29c2bb70c66f0f17271e0` |
| `a3_aligned_p1_quality.json` | `bb5df05e2ffb06997d5cb7ac30f8de8e1fc0373de8c955d3844d8a7eff90dbd1` |
| `a3_aligned_p1_quality_vs_allnpu.json` | `2fc8ffb447737b716b4112302d237cea92a193b58693b471f2c19eb68974cc13` |
| `a3_aligned_p1_c0_c4_1000.json` | `0ae7e293be538de18e69f7410f8210daac3d923a2de4ae0dd9374732ed398f14` |

## Gate

```text
R5_ALIGNED_P1 = A3_VERIFIED_READY
```

This result qualifies only the frozen TP1 P1 profile.  It does not qualify
P2, P3, TP2, Graph, Deferred Experts, Dynamic Placement, MTP, Speculative
Decoding, W8A8, or MXFP4.
