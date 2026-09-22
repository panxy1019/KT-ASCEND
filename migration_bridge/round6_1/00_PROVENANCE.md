# Round 6.1 provenance freeze

Evidence state: `CODE_INSPECTED` for Git provenance and `A3_VERIFIED` for the
execution environment. This round starts from the immutable Round 6 held-out
rejection and does not reinterpret it.

## Repository

| Item | Frozen value |
| --- | --- |
| Parent starting commit | `b4d01ee263bfbe5b5692b00adc1c85262a6487c8` |
| Round 6.1 branch | `codex/round6-1-deferred-numerical-rootcause` |
| SGLang child | `eaa58ccea4ee372b8b207dd16f398eafe3da37ff` |
| A3 source/wheel parent | `e3084d309d6f39b8a9f7185d36ac27e4b77526d7` |
| Upstream audit snapshot | `0dce4c9b097710951d1b3e232f50b973ff0a5505` |
| Freeze time (UTC) | `2026-09-15T02:01:07Z` |

The only dirty item before this branch was created was
`M third_party/llama.cpp`. The parent gitlink expects
`a94e6ff8774b7c9f950d9545baf0ce35e8d1ed2f`, while the pre-existing local
checkout is `661643e43079a4ee6faab4c1895291767b67ea8d-dirty`. Round 6.1 will not
reset, modify, stage, or commit it.

## Frozen model and D1

| Item | Value |
| --- | --- |
| Model | `deepseek-ai/DeepSeek-V2-Lite` |
| Model revision | `604d5664dddd88a0433dbae533b7fe9472482de0` |
| Tokenizer SHA256 | `41f3bf64213da8c012d8bd0871a58a1fdf70463e8f08f110ddbb1082f529f669` |
| D1 placement SHA256 | `23edad71e52c05b1dc3a227ea6e27cacad0b4db6a18c4419647efae00530d39c` |
| Round 6 GGUF SHA256 | `4b42799ec7f143569692133aadd37f089ff0c18508411e5fd37911b62c8e323f` |
| D1 layers | `{7,8}` |
| Deferred producer / terminal | `7 / 8` |
| L7 CPU experts | `{16,38,41,50}` |
| L8 CPU experts | `{8,25,33,55}` |

The historical V1 contract remains immutable:

```text
B_pair = 0.46875
K = 32
full-logit max-abs bound = 1.015625
full-logit relative-L2 bound = 0.05556806921958923
contract SHA = d5b228befe5bfc08554b7f9d1c87b4ff6d2348913d2cc94d73a9b3710ca27f26
freeze commit = 469d69aa2c872cb7a091abccbb7164c7ba6adbe9
```

## A3 environment

| Item | Value |
| --- | --- |
| Host | `110.123.0.3` |
| Container | `kt-r6-deferred-ascend` |
| Container ID | `b91b94ef502933c63247d03ae6150e3c72ce29061c4034d2c8b435b18b6d8a1f` |
| Image | `kt-r4a5-env:p2-closure-20260911` |
| Architecture | `aarch64` |
| CPU set | `0-15` |
| NPU visibility | `/dev/davinci0` only |
| CANN | `9.0.0` (`V100R001C10SPC001B250`) |
| Python | `3.11.15` |
| torch | `2.9.0+cpu` |
| torch_npu | `2.9.0.post2` |
| Host modified | `NO` |
| Business containers modified | `NO` |

Round 6.1 reuses the disposable container and persisted read-only model/GGUF
mounts so the historical case can be reproduced before instrumentation. Only
NPU0 is in scope. D2/D3 and new numerical-contract work remain forbidden.

## Evidence bundle

The machine-readable freeze is under `evidence/provenance/`. Its command file
lists the read-only commands used to establish Git and environment state. No
production patch is permitted before historical reproduction passes.
