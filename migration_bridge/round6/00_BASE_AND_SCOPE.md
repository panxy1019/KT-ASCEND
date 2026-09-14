# Round 6 Base and Scope

Evidence state: `CODE_INSPECTED`

## Repository freeze

| Item | Frozen value |
|---|---|
| GitHub evidence base | `5feb57f91aa9c786d2bc0002585a8958a456eb0f` |
| Branch | `codex/round6-deferred-expert-ascend` |
| Round 5 tag | `round5-upstream-aligned-tp1-a3-verified` |
| Tag object | `94baf6272dd86eae7e465a04f4ead630032400d4` |
| Tag target/runtime | `be5b48621db32fb894402e3e886060dea263cdc9` |
| SGLang child | `412219b4b05a888808f0b88d2963a8b6164a0fda` |

`5feb57f9` is the merge of the qualified runtime and the Round 5 evidence
branch.  A path-limited comparison confirmed no difference from the immutable
runtime tag under `kt-kernel/`, and no SGLang gitlink difference.

The pre-existing Qwen3.8 worktree is intentionally separate and was not reset,
cleaned, or used as the Round 6 source.

## Submodules

| Path | SHA |
|---|---|
| `third_party/custom_flashinfer` | `fd94393fb5b8ba8bae9c0bd6ab1c2a429d81ac76` |
| `third_party/llama.cpp` | `a94e6ff8774b7c9f950d9545baf0ce35e8d1ed2f` |
| `third_party/pybind11` | `bb05e0810b87e74709d9f4c4545f1f57a1b386f5` |
| `third_party/sglang` | `412219b4b05a888808f0b88d2963a8b6164a0fda` |

The new worktree reused local object stores for exact gitlink checkout after a
slow remote fetch was interrupted.  This affects object transport only, not
the checked-out bytes or SHAs.

## Model and immutable contracts

| Item | Frozen value |
|---|---|
| Model | `deepseek-ai/DeepSeek-V2-Lite` |
| Revision | `604d5664dddd88a0433dbae533b7fe9472482de0` |
| Tokenizer SHA256 | `41f3bf64213da8c012d8bd0871a58a1fdf70463e8f08f110ddbb1082f529f669` |
| Round 5 P3 GGUF SHA256 | `992d32e0de59e135464f77fc4869e810f3d71b3ec01f6ed0996cbeaa58e49df4` |
| Round 5 P3 placement SHA256 | `77f7d96e0b180242264f1d6eec3e4d9d16d158e9e0f9a51eac51af88e829391d` |
| Pairwise contract SHA256 | `223e738436659d389d913656952a368b6d32ccaceeef195abc2e6589c651d717` |
| Pairwise bound / K | `2.1875 / 32` |
| Quality manifest SHA256 | `c7bcb09c72f4a7213d0ecdb080f8e88e983ab1a491d4304f448bce28d8f7e1ce` |

## In scope

- DeepSeek-V2-Lite, TP1, batch/request concurrency 1, greedy, NPU0 only.
- LLAMAFILE/CPUInfer CPU experts and Ascend BF16 NPU experts.
- Static contiguous CPU-enabled layer blocks.
- `max_deferred_experts_per_token=0` and `1` only.
- Selection, pending queue, incremental merge, buffer lifetime, framework
  transfers, determinism, numerical/quality qualification and telemetry.

## Out of scope

- Dynamic Placement or online weight migration.
- `defer>=2`.
- Qwen3.8, TP2/HCCL, multi-NPU, concurrent serving, Graph, MTP,
  Speculative Decoding, W8A8, MXFP4, KML, NUMA/threadpool redesign.

## Initial gate

```text
R6_G0_PROVENANCE = COMPLETE
A3_EXECUTION = NOT_STARTED
PRODUCTION_CODE_CHANGE = SGLANG_CHILD_TOPOLOGY_GUARD
```

The first production change is isolated in SGLang child commit
`eaa58ccea4ee372b8b207dd16f398eafe3da37ff`.  It resolves the configured
Deferred budget per layer and forces every contiguous CPU-enabled block
terminal to defer zero.  It does not change routing, ownership, expert
selection, transfer, merge, or CPUInfer execution semantics.
