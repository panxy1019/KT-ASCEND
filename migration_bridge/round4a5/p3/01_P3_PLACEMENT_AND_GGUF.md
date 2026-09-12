# P3 Placement and GGUF Freeze

Evidence state: `A3_VERIFIED`

The frozen eight-layer placement was used without expert reselection:

| Layer | CPU experts | NPU expert count |
|---:|---|---:|
| 1 | 31, 43, 50, 57 | 60 |
| 5 | 14, 38, 39, 41 | 60 |
| 8 | 1, 17, 33, 45 | 60 |
| 12 | 35, 41, 46, 58 | 60 |
| 17 | 6, 8, 25, 36 | 60 |
| 19 | 14, 32, 47, 62 | 60 |
| 22 | 21, 60, 61, 62 | 60 |
| 26 | 10, 26, 30, 56 | 60 |

Frozen hashes:

| Artifact | SHA256 / size |
|---|---|
| placement canonical payload | `77f7d96e0b180242264f1d6eec3e4d9d16d158e9e0f9a51eac51af88e829391d` |
| placement JSON bytes | `c786cfffc004b64406a876cc9d370bf8fd08f7055b0e35fbf63f8d99efe79ad9` |
| placement PT bytes | `e5544faa22f210e28b02648316a139ca737aca81d9c034d757b7fd59b2e0a55b` |
| P3 F32 GGUF | `992d32e0de59e135464f77fc4869e810f3d71b3ec01f6ed0996cbeaa58e49df4` |
| P3 F32 GGUF size | `17,716,742,368` bytes |

The GGUF manifest reports model revision `604d5664`, config SHA
`f346286b0f1c8b044252fd54cb4fa78b9fab6472a6e8bebb9edfe03d414ea03d`,
hidden size 2048, MoE intermediate size 1408, 64 routed experts, and top-k 6.
The tokenizer is the frozen `DeepSeek-V2-Lite-604d5664` artifact; tokenizer
JSON SHA is `41f3bf64213da8c012d8bd0871a58a1fdf70463e8f08f110ddbb1082f529f669`.

The placement validator reported P0/P1/P2/P3 all `PASS`. Neither the corpus nor
CPU hit observations were used to alter this placement.
