# Round 4A.5 P2 Final Downstream Quality

Evidence state: `QUALITY_VERIFIED` on A3

## Provenance decision

The original Round 4A.4 quality input was recovered exactly at
`migration_bridge/round4a4/corpora/d_manifest.json`.  Its 256 inline rows,
prompt construction, expected choices, evaluator protocol, and split metadata
are versioned in the repository.  Compact canonical JSON reproduces the
frozen manifest SHA:

`c7bcb09c72f4a7213d0ecdb080f8e88e983ab1a491d4304f448bce28d8f7e1ce`.

Therefore no `quality_v2` corpus was created.  No prompt was replaced, removed,
or selected after observing P2 output.

## Environment-bound baseline rule

Historical Round 4A.4 evidence reported:

| Benchmark | historical All-NPU | historical P1 |
|---|---:|---:|
| C-Eval | 57/128 | 57/128 |
| GSM8K MC | 45/128 | 47/128 |

Those runs used Transformers `5.6.0`.  The friend PR and all Round 4A.5 P2
determinism/numerical evidence were produced with Transformers `5.3.0.dev0`.
Reproducing All-NPU with the exact frozen rows in the latter environment gave
different absolute scores.  The row IDs and choice token IDs were checked;
the difference is an execution-environment boundary, not missing input bytes.

It would be invalid to compare a current P2 result against the old 57/45
baseline or to switch Transformers after the P2 numerical qualification.  The
formal P2 A/B therefore uses newly reproduced All-NPU and P2 results in the
same Round 4A.5 container, model, tokenizer, prompt set, and evaluator.  The
old results remain historical evidence and are not retroactively changed.

## Frozen protocol

| Field | Value |
|---|---|
| C-Eval | 128 frozen examples |
| GSM8K MC | 128 frozen examples |
| decoding | greedy, temperature 0 |
| protocol seed | 0 |
| model snapshot | `DeepSeek-V2-Lite-604d5664` |
| config SHA | `f346286b0f1c8b044252fd54cb4fa78b9fab6472a6e8bebb9edfe03d414ea03d` |
| tokenizer JSON SHA | `41f3bf64213da8c012d8bd0871a58a1fdf70463e8f08f110ddbb1082f529f669` |
| SGLang | `cbaa79c2f5b004ab4ca470c9fa56161666ccafb2` |
| Transformers | `5.3.0.dev0` |
| P2 placement | layers `{1,9,17,26}`, 16 CPU experts |
| placement canonical SHA | `f6d4e9c6a2e5e8060e846dbc7c628d069c9aa6150aaa1e2690ea28a51ba286a3` |

## Paired A/B results

| Benchmark | All-NPU | Hybrid P2 | Absolute delta | invalid NPU/P2 | paired bootstrap 95% CI | Decision |
|---|---:|---:|---:|---:|---:|---:|
| C-Eval | 27/128 (`0.2109375`) | 27/128 (`0.2109375`) | `0` | `0 / 0` | `[0, 0]` | no regression |
| GSM8K MC | 31/128 (`0.2421875`) | 31/128 (`0.2421875`) | `0` | `0 / 0` | `[0, 0]` | no regression |

All output token sequences passed the evaluator's validity checks.  At the
per-example correctness level there were zero discordant pairs in either
benchmark:

```text
All-NPU correct / P2 wrong = 0
All-NPU wrong / P2 correct = 0
```

The paired bootstrap used 20,000 repeats and seed 0.  This supports the frozen
quality conclusion `NO_STATISTICALLY_MEANINGFUL_REGRESSION`; it is not a claim
that all logits or all free-running output bytes are bitwise equal.

## Evidence

| Artifact | Internal canonical SHA | File SHA |
|---|---|---|
| All-NPU raw result | `ff54b390d24e48b8075ac8003045f038a5e27d0281d68ab37977ac14930ed950` | `8aeb284cd055d67bacd3bfc069cdc59ee5bbfa2521fc1b2981c52847aad20dc7` |
| P2 raw result | `4959a88f123a3455b19818606aad9cfa9d3825a1f53d1602000c5184681c55d0` | `7003a378aa200ed2f9ab18a1b502286b7a687d45456ca057f008e829bc184ef8` |
| paired summary | `6d3b4ba44c67a8df399bfcd76a38de17204390f27a893a893191e449ba3bf807` | `2fc8ffb447737b716b4112302d237cea92a193b58693b471f2c19eb68974cc13` |

Compact files are under `evidence/final-p2/`; server logs remain in the A3
artifact root `/home/admin/kt-artifacts/round4a5-closure`.

## Decision

```text
P2_QUALITY_INPUT_PROVENANCE = CLOSED
P2_DOWNSTREAM_QUALITY       = A3_VERIFIED_READY
```
