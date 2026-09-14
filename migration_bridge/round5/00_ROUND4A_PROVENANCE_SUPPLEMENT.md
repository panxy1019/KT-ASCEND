# Round 5 — Round 4A Provenance Supplement

Evidence state: `CODE_INSPECTED` plus artifact hashes reverified on A3.

## Immutable local reference

| Item | Frozen value |
|---|---|
| Round 4A tag | `round4a-tp1-multiplacement-a3-verified` |
| annotated tag object | `9e2c9e220337f57b55f1fe9b50da0862ccc0cf64` |
| dereferenced target | `096a6ef860a01dc0a3c8f2cc5618a396425ee09b` |
| tag kind | annotated, unsigned |
| P2 immutable tag | `round4a5-p2-a3-verified` |
| P2 immutable target | `d4830525c6806d5d587b3096634614aad7fef181` |
| Round 4A SGLang child | `cbaa79c2f5b004ab4ca470c9fa56161666ccafb2` |

The tag object and peeled target were independently resolved locally and from
`panxy1019/KT-ASCEND`. The tag remains unchanged. Round 5 audit branch
`feature/kt-round5-alignment-audit` was created from the peeled target; no
Round 4A branch or tag was rebased.

## Full model and validation provenance

| Item | Frozen value |
|---|---|
| model repository | `deepseek-ai/DeepSeek-V2-Lite` |
| full model revision | `604d5664dddd88a0433dbae533b7fe9472482de0` |
| previously used short revision | `604d5664` |
| config SHA256 | `f346286b0f1c8b044252fd54cb4fa78b9fab6472a6e8bebb9edfe03d414ea03d` |
| tokenizer JSON SHA256 | `41f3bf64213da8c012d8bd0871a58a1fdf70463e8f08f110ddbb1082f529f669` |
| P3 GGUF SHA256 | `992d32e0de59e135464f77fc4869e810f3d71b3ec01f6ed0996cbeaa58e49df4` |
| P3 placement canonical SHA256 | `77f7d96e0b180242264f1d6eec3e4d9d16d158e9e0f9a51eac51af88e829391d` |
| pairwise contract canonical SHA256 | `223e738436659d389d913656952a368b6d32ccaceeef195abc2e6589c651d717` |
| quality manifest canonical SHA256 | `c7bcb09c72f4a7213d0ecdb080f8e88e983ab1a491d4304f448bce28d8f7e1ce` |

The A3 snapshot directory reproduced the config and tokenizer hashes. The full
40-character model revision supplements earlier short-form documentation; it
does not change the bytes qualified in Round 4A.

## Pinned upstream candidate

| Item | Frozen value |
|---|---|
| KTransformers | `440df948027128318bc30a11a545433eb0c7d383` |
| SGLang gitlink | `3424f35d0e60f03b9e75d9b90941f8f4cb944524` |
| official Ascend integration commit | `040a498` (`#2157`) |
| candidate branch | `feature/kt-round5-upstream-alignment-dsv2lite` |

At freeze time, upstream KTransformers `main` resolved exactly to the pinned
parent. Upstream SGLang `main` had already advanced beyond the pinned gitlink,
so Round 5 uses the gitlink commit and never the dynamic child tip.

## Dual checkout state

| Checkout | Path | State |
|---|---|---|
| local reference | `/home/admin/Desktop/KT_ASCEND/kt-local-r4a-reference` | detached at `096a6ef...`; no local changes |
| upstream candidate | `/home/admin/Desktop/KT_ASCEND/kt-upstream-r5` | branch from `440df948...`; SGLang at `3424f35...` |

The first full-recursive local clone exhausted the 32 GB workstation root
filesystem while checking out the out-of-scope ROCm
`custom_flashinfer/3rdparty/composable_kernels` tree. That failed, unmodified
clone was moved intact to `/dev/shm/kt-upstream-r5-failed-recursive` so no
pre-existing project data was deleted. The replacement filtered checkout
initializes SGLang, llama.cpp, and pybind11 and records the uninitialized
custom_flashinfer gitlink `fd94393...`. A full recursive checkout will be
re-attempted in the new A3 disposable environment before raw upstream build;
the local omission is not counted as `R5_UPSTREAM_BUILD=PASS`.

## Gate

```text
R5_G0_PROVENANCE_COMPLETE = PASS
ROUND4A_TAG_MUTATED        = NO
UPSTREAM_PARENT_PINNED     = YES
UPSTREAM_SGLANG_PINNED     = YES
```
