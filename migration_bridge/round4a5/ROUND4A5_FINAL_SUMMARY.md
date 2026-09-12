# Round 4A.5 Final Summary

## Final decision

All frozen P2 gates have closed on A3.  Raw ACL pointer-based D2H/H2D was
replaced by framework-managed Tensor `copy_`; the cleaned production path is
deterministic across repeats and process restarts, passes the unchanged
pairwise contract, covers all selected CPU experts, and shows no paired quality
regression.

```text
ROUND4A5_ROOT_CAUSE            = ATTRIBUTED
ASCEND_FRAMEWORK_TRANSFER_FIX  = A3_VERIFIED_READY
P2_SAME_PATH_DETERMINISM       = A3_VERIFIED_READY
P2_CLEAN_RESTART               = A3_VERIFIED_READY
P2_PAIRWISE_NUMERICAL          = A3_VERIFIED_READY
P2_CPU_COVERAGE                = A3_VERIFIED_READY
P2_CPU_NOT_HIT                 = A3_VERIFIED_READY
P2_DOWNSTREAM_QUALITY          = A3_VERIFIED_READY
P2_RETAINED_REGRESSION         = PASS
P2_EVIDENCE_PROVENANCE         = CLOSED
MULTI_LAYER_P2                 = A3_VERIFIED_READY
P3_ALLOWED                     = YES
```

## Repository freeze

| Field | Value |
|---|---|
| PR takeover base | `305b84359d6a646ab260a16d004d19147103d561` |
| PR base | `2ae5523ce1d1b30f60baf78917af1229f57a34aa` |
| local closure branch | `feature/kt-round4a5-p2-closure` |
| A3-verified runtime/test candidate | `d36dd8f3e01c9a86d87309e972b11e75b537b68e` |
| final P2 tag | `round4a5-p2-a3-verified` |
| final P2 tag target | `TO_BE_RECORDED_AFTER_IMMUTABLE_TAG_CREATION` |
| SGLang child | `cbaa79c2f5b004ab4ca470c9fa56161666ccafb2` |

No upstream KTransformers rebase, merge, or source substitution occurred.
The friend handoff tag `round4a5-friend-takeover-305b843` remains fixed at the
exact PR head.

## A3 environment

| Item | Value |
|---|---|
| environment ID | `a3-r4a5-p2-closure-0825d8eb90af` |
| host CPU | Kunpeng 920 7280Z, aarch64 |
| host NUMA | 8 nodes |
| assigned CPU | cpuset `0-15` |
| assigned accelerator | NPU0 only |
| disposable container | `kt-r4a5-p2-closure-env` |
| container ID | `0825d8eb90afcba292433acc9230780c575f86a5853b34db01d1141ee36ff209` |
| image | `sha256:7781588ef508d2cf37bb4a09878b0079bf13795fedd874cd7c76c4da47f2ea3a` |
| Python | `3.11.15` |
| torch | `2.9.0+cpu` |
| torch_npu | `2.9.0.post2` |
| Transformers | `5.3.0.dev0` |
| tokenizers | `0.22.2` |
| NumPy | `1.26.4` |
| host modified | NO |
| business container modified | NO |
| NPU1–7 used | NO |

The P2 service was stopped after qualification.  Port 31000 was released and
the final `npu-smi` check reported no running process on NPU0–7.

## Model, artifacts, and contracts

| Item | SHA / value |
|---|---|
| model snapshot | `DeepSeek-V2-Lite-604d5664` |
| model config SHA | `f346286b0f1c8b044252fd54cb4fa78b9fab6472a6e8bebb9edfe03d414ea03d` |
| tokenizer JSON SHA | `41f3bf64213da8c012d8bd0871a58a1fdf70463e8f08f110ddbb1082f529f669` |
| P2 F32 GGUF SHA | `2c4cd307a53b761c4191cf108d1e00ec2bd2c99b48a595a28ac8fdd0e0edd1fa` |
| P2 placement | layers `{1,9,17,26}`, 16 CPU experts |
| P2 placement canonical SHA | `f6d4e9c6a2e5e8060e846dbc7c628d069c9aa6150aaa1e2690ea28a51ba286a3` |
| pairwise contract version | `round4a4-v1` |
| formal contract canonical SHA | `223e738436659d389d913656952a368b6d32ccaceeef195abc2e6589c651d717` |
| candidate contract SHA used by validator | `accbbc15baf49d869b062dc2b71b4fe6f71a7fa8bd74f27120757d1c1f0e6627` |
| `B_pair` / candidate K | `2.1875` / `32` |
| quality manifest canonical SHA | `c7bcb09c72f4a7213d0ecdb080f8e88e983ab1a491d4304f448bce28d8f7e1ce` |

The formal contract file embeds the candidate contract SHA.  P2's validation
artifacts report the candidate SHA because that is the frozen numerical rule
payload consumed by the validator; this does not indicate a refit or a
different `B_pair`.

## Root cause and production fix

The first observed P2 same-path divergence was at Layer 26 CPU output while
its input, router IDs, routing weights, and NPU partial were exact.  Negative
controls rejected shared CPUInfer, shared LLAMAFILE scratch, TP merge storage,
worker count, and a D2H destination clone as sufficient causes.  The decisive
A/B was:

```text
raw acl.rt.memcpy pointer transfer -> nondeterministic
framework-managed Tensor copy_    -> deterministic
```

Production fix commit:

`00c7acfe3569a6a7a6af5b9891e9989e334dffb3`

The normal Ascend CPU expert path now uses blocking framework-managed copies
for D2H and H2D while retaining host tensors until CPUInfer completion.  An
explicit ACL stream synchronization remains an ordering primitive; raw ACL
pointer memcpy is not used for the data transfers.  Routing, mapping, shared
expert ownership, routed scaling, weights, and CPU arithmetic were not changed.

## Final P2 gate table

| Gate | Final state | Evidence |
|---|---|---|
| Root cause | ATTRIBUTED | `05_RAW_ACL_TRANSFER_ROOT_CAUSE.md` |
| Production transfer fix | PASS | commit `00c7acfe...`; `08_PR_CLEANUP_AUDIT.md` |
| Same-process determinism | PASS | 9 prompts × 10 repeats × 64; `evidence/final-p2/same-process-validation9.json` |
| Clean restart | PASS | 9/9 exact; `evidence/final-p2/clean-restart-compare-validation9.json` |
| Pairwise numerical | PASS | Q2/H2, 512 positions, zero overflow |
| CPU coverage | PASS | 4/4 layers, 16/16 experts, 74,807 hits |
| CPU-not-hit | PASS | 32/32 selected real rows exact |
| Downstream quality | PASS | 128+128, zero paired correctness delta |
| Retained regression | PASS | Round2A/2B/2C, SGLang, Round3, P1 compact |
| Lifecycle stability | PASS | P2 5,760 generated tokens; 1,000 CPU and 1,000 transfer cycles |
| Evidence provenance | CLOSED | exact PR/child/model/GGUF/placement/contract/quality hashes |

## Determinism and transfer behavior

Same-process:

```text
9 prompts × 10 repeats × 64 tokens
all_repeat_exact = true
all_prefix_exact at 8/16/32 = true
```

Clean restart:

```text
before process internally exact = true
after process internally exact  = true
9/9 reference trajectories      = exact
```

Internal canonical evidence SHAs:

| Evidence | SHA |
|---|---|
| same process | `b4bab66850b2dc6871627bb06040ae62f6a9351ecb02f16d3414d17c9ab133a7` |
| restarted process | `6fdad6d47d4fe3f2df5ae50d2267c8664a5a4b5db2afd3d7bc6226c8ee861f5b` |
| restart comparison | `2f7003a908732b01c9d7a07aaf17af4b02fe0973f8d65acca301682b6779e559` |

## Frozen pairwise numerical result

| Corpus | Stable | Stable exact | Ambiguous | Membership | Overflow | Finite |
|---|---:|---:|---:|---:|---:|---:|
| Q2 | 137 | 137 | 119 | 119 | 0 | yes |
| H2 | 155 | 155 | 101 | 101 | 0 | yes |

Canonical result SHAs are `c9f7b98f...` for Q2 and `44cbac8a...` for H2.
The bound and K were not changed after P1 or during P2.

## Coverage and controls

All selected layers and experts were exercised:

| Layer | selected/exercised | route hits |
|---:|---:|---:|
| 1 | 4/4 | 17,890 |
| 9 | 4/4 | 20,076 |
| 17 | 4/4 | 20,974 |
| 26 | 4/4 | 15,867 |

Total CPU route hits were 74,807 and all routing weights were finite.  The
CPU-not-hit control found 1,154 eligible real Layer-17 positions and verified
the frozen minimum of 32: CPU output was exact zero and Hybrid merge exactly
equalled the NPU partial in every selected position.

## Downstream quality

The original frozen quality input was recovered exactly.  Under the common
Round 4A.5 environment:

| Benchmark | All-NPU | P2 | Delta | paired bootstrap 95% CI |
|---|---:|---:|---:|---:|
| C-Eval | 27/128 | 27/128 | 0 | `[0,0]` |
| GSM8K MC | 31/128 | 31/128 | 0 | `[0,0]` |

Invalid outputs were zero.  Historical 57/45 absolute scores belong to a
Transformers 5.6.0 environment and were not mixed into this paired decision.
See `11_P2_FINAL_QUALITY.md`.

## Regression and stability

Fresh results at the closure candidate:

```text
Round 2A: 21 passed
Round 2B: 23 passed, 1 expected CUDA-only skip
Round 2C: 44 passed
SGLang KT EP: 4 passed
Round 3 real-model compact: PASS
Round 4A.4 P1 C0-C4 + 1000-forward: PASS
```

The stale Round 2A BF16-return assertions were corrected to the preserved FP32
partial contract in a test-only commit.  Round 2B's output buffer is FP32 and
its former overwrite symptom did not recur.  No unexplained regression remains.

## P3 entry freeze

The pre-existing P3 placement has been inspected but not yet executed:

```text
layers = {1,5,8,12,17,19,22,26}
CPU experts = 32
placement canonical SHA = 77f7d96e0b180242264f1d6eec3e4d9d16d158e9e0f9a51eac51af88e829391d
placement JSON file SHA = c786cfffc004b64406a876cc9d370bf8fd08f7055b0e35fbf63f8d99efe79ad9
placement PT file SHA   = e5544faa22f210e28b02648316a139ca737aca81d9c034d757b7fd59b2e0a55b
```

P3 must branch from the immutable P2 tag, retain framework-managed transfers,
and use the same `B_pair=2.1875`, K=32, quality manifest, model, tokenizer, and
runtime semantics.  Upstream alignment remains explicitly out of scope.
