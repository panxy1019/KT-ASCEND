# Round 6.1 final summary

## Scope and provenance

Round 6.1 investigated the frozen D1 Deferred numerical rejection. It did not
change placement, weights, the old threshold, QD/HD corpora, or production
semantics. It did not execute D2 or D3.

| Item | Value |
| --- | --- |
| Investigation branch | `codex/round6-1-deferred-numerical-rootcause` |
| Frozen Round 6 rejection start | `b4d01ee263bfbe5b5692b00adc1c85262a6487c8` |
| Historical contract freeze commit | `469d69aa2c872cb7a091abccbb7164c7ba6adbe9` |
| A3 source/wheel parent | `e3084d309d6f39b8a9f7185d36ac27e4b77526d7` plus hash-frozen diagnostic overlay |
| SGLang base on A3 | `eaa58ccea4ee372b8b207dd16f398eafe3da37ff` plus hash-frozen diagnostic overlay |
| Local diagnostic SGLang commit | `35b12430806221ca01574f550a9b7f9eb8b4cf65` |
| Model revision | `604d5664dddd88a0433dbae533b7fe9472482de0` |
| D1 placement SHA256 | `23edad71e52c05b1dc3a227ea6e27cacad0b4db6a18c4419647efae00530d39c` |
| GGUF SHA256 | `4b42799ec7f143569692133aadd37f089ff0c18508411e5fd37911b62c8e323f` |
| Container | `kt-r6-deferred-ascend` / `b91b94ef5029...` |
| NPU visibility | NPU0 only |
| CANN / torch / torch_npu | 9.0.0 / 2.9.0+cpu / 2.9.0.post2 |
| Host modified | NO |
| Business containers modified | NO |

The A3 checkout intentionally remained at the frozen production parent/child
and received only diagnostic file overlays. SHA256 values for every overlay
match the committed local files; the raw tensor checksum indexes preserve the
executed evidence identity.

## Decisive result

Production Deferred is a faithful implementation of the audited intended
semantics for the frozen failure:

```text
production B vs explicit replay C:
  aligned stage tensors = 436/436 exact
  final logits          = 102400/102400 exact
  final SHA256          = 98177a8cb38a920d0f70ead9ae60b947a9a56ed4342fbc16933fb32f89145c33
```

The first A/B difference is the L7 CPU partial intentionally made visible at
L8. Later layers amplify it to full-logit max abs 0.8125 and pairwise distortion
0.75, while top-1 remains 21.

Runtime/lifetime checks and the whole-request CPU-not-hit control passed. No
implementation invariant failure was found.

## Gate matrix

| Gate | State | Evidence |
| --- | --- | --- |
| Provenance | PASS | `00_PROVENANCE.md` |
| Upstream semantic audit | PASS | `01_UPSTREAM_DEFERRED_SEMANTIC_AUDIT.md` |
| Historical reproduction | PASS | `02_HISTORICAL_CASE_REPRODUCTION.md` |
| Stagewise localization | PASS | `02_STAGEWISE_LOCALIZATION.md` |
| Explicit replay oracle | PASS | `03_EXPLICIT_DEFERRED_REPLAY_ORACLE.md` |
| Runtime/lifetime controls | PASS | `04_RUNTIME_LIFETIME_CONTROLS.md` |
| Whole-request CPU-not-hit | PASS | `05_WHOLE_REQUEST_CPU_NOT_HIT_CONTROL.md` |
| Descriptive numerical risk | COMPLETE | `06_NUMERICAL_RISK_ANALYSIS.md` |
| Root-cause ADR | INTENDED_DEFERRED_APPROXIMATION | `07_ROOT_CAUSE_ADR.md` |

## Machine-readable status

```text
ROUND6_1_HISTORICAL_REPRODUCTION          = PASS
ROUND6_1_UPSTREAM_SEMANTIC_AUDIT          = PASS
ROUND6_1_STAGEWISE_LOCALIZATION           = PASS
ROUND6_1_EXPLICIT_REPLAY_ORACLE           = PASS
ROUND6_1_RUNTIME_LIFETIME_CONTROLS         = PASS
ROUND6_1_WHOLE_REQUEST_CPU_NOT_HIT_CONTROL = PASS

ROUND6_1_ROOT_CAUSE =
INTENDED_DEFERRED_APPROXIMATION

ROUND6_V1_CONTRACT =
HISTORICAL_HELDOUT_REJECTED

ROUND6_D1 =
BLOCKED_PENDING_NEW_INDEPENDENT_CONTRACT

ROUND6_D2 = NOT_RUN
ROUND6_D3 = NOT_RUN
```

Round 6.1 is complete at Gate A. The next permissible step is an explicitly
authorized Round 6.2 with a precommitted independent numerical mechanism and a
new held-out corpus. It is not permissible to continue D2/D3 now.
