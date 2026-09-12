# Round 4A Final Provenance

## Repository

| Field | Frozen value |
|---|---|
| repository | `panxy1019/KT-ASCEND` |
| P2 immutable source | `round4a5-p2-a3-verified^{}` = `d4830525c6806d5d587b3096634614aad7fef181` |
| P3 branch | `feature/kt-round4a5-p3` |
| SGLang child | `cbaa79c2f5b004ab4ca470c9fa56161666ccafb2` |
| final Round 4A tag | `round4a-tp1-multiplacement-a3-verified` |

The A3 runtime checkout's production tree was byte-equivalent to the P2 tag.
Tracked differences were limited to retained tests and evidence/docs; runtime
generated files were untracked. The SGLang child contained only an untracked
`fusion_result.json`. No claim relies on a dirty production source file.

## Model and data

| Field | Frozen value |
|---|---|
| model | DeepSeek-V2-Lite |
| model revision | `604d5664` |
| config SHA | `f346286b0f1c8b044252fd54cb4fa78b9fab6472a6e8bebb9edfe03d414ea03d` |
| tokenizer JSON SHA | `41f3bf64213da8c012d8bd0871a58a1fdf70463e8f08f110ddbb1082f529f669` |
| P3 GGUF SHA | `992d32e0de59e135464f77fc4869e810f3d71b3ec01f6ed0996cbeaa58e49df4` |
| P3 GGUF bytes | `17,716,742,368` |
| P3 placement canonical SHA | `77f7d96e0b180242264f1d6eec3e4d9d16d158e9e0f9a51eac51af88e829391d` |
| numerical contract canonical SHA | `223e738436659d389d913656952a368b6d32ccaceeef195abc2e6589c651d717` |
| quality manifest canonical SHA | `c7bcb09c72f4a7213d0ecdb080f8e88e983ab1a491d4304f448bce28d8f7e1ce` |

## A3 execution environment

Environment `a3-r4a5-p3-0825d8eb90af` used container
`0825d8eb90afcba292433acc9230780c575f86a5853b34db01d1141ee36ff209`
on Kunpeng 920 with cpuset 0-15 and only `/dev/davinci0`. CANN 9.0.0,
Python 3.11.15, torch 2.9.0+cpu, torch_npu 2.9.0.post2, Transformers
5.3.0.dev0, and tokenizers 0.22.2 were frozen throughout P3.

The host system and unrelated business containers were not modified. NPU1-7
were not mapped to the disposable container. Final serving shutdown released
the assigned device.

## Evidence inventory

Reviewable evidence is committed in `round4a5/p3/evidence/`; byte hashes are
listed in `SHA256SUMS`. Large raw tensors and server logs remain under:

`/home/admin/kt-artifacts/round4a5-closure/`

on A3. The committed evidence includes all gate decisions, both stability raw
campaign JSONs and memory time series, quality rows, and retained regression
logs. The GGUF itself is identified by size and SHA rather than committed.
