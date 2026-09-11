# Round 4A.5 Takeover Baseline

Status: `TAKEOVER_PROVENANCE_VERIFIED`

Evidence state: `CODE_INSPECTED`

Date: 2026-09-11 UTC

## Repository and PR

| Field | Frozen value |
|---|---|
| Repository | `panxy1019/KT-ASCEND` |
| PR | `#1` |
| PR URL | `https://github.com/panxy1019/KT-ASCEND/pull/1` |
| PR state at takeover | `OPEN`, `MERGEABLE`, merge state `CLEAN` |
| PR base branch | `feature/kt-round4a4-pairwise-margin-qualification` |
| PR base SHA | `2ae5523ce1d1b30f60baf78917af1229f57a34aa` |
| Friend fork | `Re1lya/KT-ASCEND` |
| Friend branch | `feature/kt-round4a5-p2-determinism` |
| Expected friend head | `305b84359d6a646ab260a16d004d19147103d561` |
| Resolved friend head | `305b84359d6a646ab260a16d004d19147103d561` |
| Resolved PR ref | `305b84359d6a646ab260a16d004d19147103d561` |
| Local continuation branch | `feature/kt-round4a5-p2-closure` |
| Local safety tag | `round4a5-friend-takeover-305b843` |

The friend branch and the direct PR ref were fetched independently:

```text
re1lya/feature/kt-round4a5-p2-determinism
= 305b84359d6a646ab260a16d004d19147103d561

kt-ascend/pr/1
= 305b84359d6a646ab260a16d004d19147103d561
```

There was no head mismatch. The continuation branch was created directly from
that commit without rebasing, squashing, cherry-picking, or merging upstream
KTransformers.

## Remote mapping

The existing local clone uses different remote names from the generic takeover
guide:

| Local remote | URL | Role |
|---|---|---|
| `kt-ascend` | `https://github.com/panxy1019/KT-ASCEND.git` | project repository and PR ref |
| `re1lya` | `https://github.com/Re1lya/KT-ASCEND.git` | friend fork |
| `origin` | `https://github.com/kvcache-ai/ktransformers.git` | official upstream, inspection only |

The PR was therefore fetched from `kt-ascend pull/1/head`, not from `origin`.
No official-upstream update was incorporated.

## Parent worktree state

Before takeover:

```text
branch = feature/kt-round4a4-pairwise-margin-qualification
HEAD   = 2ae5523ce1d1b30f60baf78917af1229f57a34aa
status = clean
```

After takeover:

```text
branch       = feature/kt-round4a5-p2-closure
HEAD         = 305b84359d6a646ab260a16d004d19147103d561
git describe = round4a5-friend-takeover-305b843
status       = clean
```

The annotated safety tag targets the immutable friend head and carries the
message `Round 4A.5 friend handoff baseline before P2 closure`. It must not be
moved to a later cleanup or qualification commit.

## Submodule freeze

`git submodule sync --recursive` and `git submodule update --init --recursive`
completed without changing the parent worktree. Every submodule was clean.

| Submodule | Frozen SHA |
|---|---|
| `third_party/custom_flashinfer` | `fd94393fb5b8ba8bae9c0bd6ab1c2a429d81ac76` |
| `third_party/custom_flashinfer/3rdparty/composable_kernels` | `5055b3bdcb5a7f0b8f359b606d3c5b75efd6df54` |
| `third_party/custom_flashinfer/3rdparty/cutlass` | `cc3c29a81a140f7b97045718fb88eb0664c37bd7` |
| `third_party/custom_flashinfer/3rdparty/googletest` | `5a37b517ad4ab6738556f0284c256cae1466c5b4` |
| `third_party/custom_flashinfer/3rdparty/mscclpp` | `cddffbc8b6dfa6facf7c64c1b7d73acf30e600b3` |
| `third_party/custom_flashinfer/3rdparty/nvbench` | `555d628e9b250868c9da003e4407087ff1982e8e` |
| `third_party/custom_flashinfer/3rdparty/spdlog` | `c3aed4b68373955e1cc94307683d44dca1515d2b` |
| `third_party/llama.cpp` | `a94e6ff8774b7c9f950d9545baf0ce35e8d1ed2f` |
| `third_party/llama.cpp/kompute` | `4565194ed7c32d1d2efa32ceab4d3c6cae006306` |
| `third_party/pybind11` | `bb05e0810b87e74709d9f4c4545f1f57a1b386f5` |
| `third_party/sglang` | `cbaa79c2f5b004ab4ca470c9fa56161666ccafb2` |

SGLang audit:

```text
origin = https://github.com/kvcache-ai/sglang.git
branch = feature/kt-ep-round4a4-pairwise-margin-qualification
HEAD   = cbaa79c2f5b004ab4ca470c9fa56161666ccafb2
status = clean
```

The child was not advanced to any remote tip.

## PR delta against the frozen Round 4A.4 base

At takeover the PR contains 39 changed files, 5,023 additions, and 46
deletions. Its history preserves reproduction, layer bisection, negative
controls, root-cause attribution, the production transfer fix, qualification
tool compatibility, and interim A3 evidence.

The production fix is isolated in:

```text
00c7acfe3569a6a7a6af5b9891e9989e334dffb3
fix(ascend): use framework-managed CPU expert transfers
```

The final friend-head commit is:

```text
305b84359d6a646ab260a16d004d19147103d561
docs(round4a5): record P2 fix requalification
```

## Frozen contracts carried into closure

| Contract | Frozen value |
|---|---|
| Model revision | `604d5664` |
| SGLang child | `cbaa79c2f5b004ab4ca470c9fa56161666ccafb2` |
| P2 CPU layers | `{1,9,17,26}` |
| P2 selected CPU experts | 16 total, four per selected layer |
| P2 placement SHA256 | `f6d4e9c6a2e5e8060e846dbc7c628d069c9aa6150aaa1e2690ea28a51ba286a3` |
| P2 GGUF SHA256 | `2c4cd307a53b761c4191cf108d1e00ec2bd2c99b48a595a28ac8fdd0e0edd1fa` |
| Pairwise contract | `round4a4-v1` |
| `B_pair` | `2.1875` |
| Candidate K | `32` |
| Pairwise contract SHA256 | `223e738436659d389d913656952a368b6d32ccaceeef195abc2e6589c651d717` |
| Historical quality manifest SHA256 | `c7bcb09c72f4a7213d0ecdb080f8e88e983ab1a491d4304f448bce28d8f7e1ce` |

None of these values is authorized to change silently during P2 closure.

## Takeover decision

```text
PR_HEAD_MATCH                 = YES
BASE_SHA_MATCH                = YES
PARENT_WORKTREE_CLEAN         = YES
SUBMODULES_REPRODUCIBLE       = YES
SGLANG_CHILD_CLEAN            = YES
UPSTREAM_REBASE_OR_MERGE      = NO
TAKEOVER_PROVENANCE           = CLOSED
```

The next allowed actions are diagnostic cleanup review and frozen-quality input
provenance recovery. This document does not promote any local or historical
evidence to a new A3 qualification result.
