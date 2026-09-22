# Round 6 Deferred Expert Ascend — Interim Summary

## Outcome

Round 6 has reached its first held-out numerical gate and stopped there.
DeepSeek-V2-Lite Deferred Expert execution is implemented and deterministic
for frozen D1, but the QD-derived pairwise numerical candidate did not
generalize to HD. Accordingly D1 is not promoted and D2/D3 were not run.

## Repository and A3 artifact under test

| Item | Value |
| --- | --- |
| Branch | `codex/round6-deferred-expert-ascend` |
| A3 source/wheel parent | `e3084d309d6f39b8a9f7185d36ac27e4b77526d7` |
| SGLang child | `eaa58ccea4ee372b8b207dd16f398eafe3da37ff` |
| Production parent fix | `13a0c8679` — consume pending state exactly once |
| Production child fix | `eaa58ccea4` — guard Deferred block terminals |
| A3 container | `kt-r6-deferred-ascend` |
| NPU | NPU0 only |
| Host/business container mutation | `NO` |

The later parent commits are test tooling, protocol and evidence commits. The
pre-existing dirty `third_party/llama.cpp` worktree state was preserved and
was not reset or included in Round 6 commits.

## Frozen configuration

| Item | Value |
| --- | --- |
| Model revision | `604d5664dddd88a0433dbae533b7fe9472482de0` |
| Round 6 GGUF SHA | `4b42799ec7f143569692133aadd37f089ff0c18508411e5fd37911b62c8e323f` |
| D1 placement SHA | `23edad71e52c05b1dc3a227ea6e27cacad0b4db6a18c4419647efae00530d39c` |
| D1 layers | `{7,8}`; producer 7, terminal 8 |
| CPU experts | L7 `{16,38,41,50}`, L8 `{8,25,33,55}` |
| QD SHA | `fe28d7fe4b4421b53901950b41a7d98b58650de359bd5141e9d05ef71e6bf6f1` |
| HD SHA | `0ee3b50c9d02515f1492463ea1af5d507ad72267d43d5e0962e6bf9c6619adba` |

## Completed evidence

- Local source/build tests: parent 7/7 and child 7/7.
- Retained defer=0 compact serving: P1_B0/P2_B0/P3_B0, four prompt
  categories, two repeats, 32 tokens and prefix checks: PASS. This was a
  compact baseline, not the complete retained WP7 suite.
- Real-weight incremental merge: Layer 7 split and Layer 8 incremental output
  exact under the production boundary; 100 repeats one hash.
- Exactly-once trace: 804 records, no false/missed pending, final pending set
  empty.
- D1 same-process determinism: 12 prompts × 10 repeats × 64 tokens plus
  8/16/32 prefixes: PASS.
- D1 clean restart: 12/12 trajectories and input-token hashes exact across
  independent serving processes: PASS.
- QD matched-history sweep: 1024/1024 positions complete, finite, 79 actual
  Deferred target-row positions.
- QD-only candidate: `B_pair=0.46875`, stable `855/855`, ambiguous membership
  `169/169`, zero QD overflow. Candidate canonical SHA:
  `d5b228befe5bfc08554b7f9d1c87b4ff6d2348913d2cc94d73a9b3710ca27f26`.
- HD matched-history sweep: 1024/1024 positions complete, finite, 102 actual
  Deferred target-row positions.

## Current blocker

HD produced one pairwise envelope overflow:

```text
hd_en_02, token 43
observed pairwise distortion = 0.75
frozen bound = 0.46875
```

All 888 stable positions retained exact top-1 and all 136 ambiguous positions
passed membership. Full-logit max-absolute and relative-L2 bounds did not
overflow. Nevertheless the contract is rejected because zero pairwise
overflow was predeclared as a hard gate before HD.

The QD/HD corpora naturally contained no full request with zero CPU routes;
target-row zero-hit observations still had CPU routes in prompt-prefix rows.
Thus the independent whole-request CPU-not-hit gate remains `NOT_OBSERVED`,
not PASS and not the cause of the current rejection.

## Gate matrix

| Gate | State |
| --- | --- |
| Provenance / frozen corpora / placement / GGUF | PASS |
| Deferred production mechanism | A3 exercised |
| D1 exactly-once and pending drain | PASS |
| D1 same-process determinism | PASS |
| D1 clean restart | PASS |
| QD numerical derivation | PASS — candidate frozen |
| HD numerical held-out | **REJECTED — one pairwise overflow** |
| Whole-request CPU-not-hit exact control | NOT_OBSERVED |
| FD free generation | NOT_RUN by stop rule |
| Downstream quality | NOT_RUN by stop rule |
| D1 full promotion | BLOCKED |
| D2 | NOT_RUN |
| D3 | NOT_RUN |

## Honest project status

```text
ROUND5_UPSTREAM_ALIGNMENT = A3_VERIFIED_READY  (unchanged baseline)
ROUND6_DEFERRED_IMPLEMENTATION = D1_A3_EXERCISED
ROUND6_D1_DETERMINISM = A3_VERIFIED_READY
ROUND6_D1_NUMERICAL_ACCEPTANCE = HELDOUT_REJECTED
ROUND6_D1 = BLOCKED_AT_HD_PAIRWISE_ENVELOPE
ROUND6_D2 = NOT_RUN
ROUND6_D3 = NOT_RUN
ROUND6_DEFERRED_EXPERT_ASCEND = BLOCKED
```

Round 5 and earlier verified capabilities are not downgraded. No GitHub push
or main-branch merge is implied by this interim summary; that remains a
separate explicit repository operation.
