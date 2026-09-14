# P2 Resume

Status: `BLOCKED`

P2 used the frozen Round 4A placement without expert reselection:

- CPU-enabled layers: `{1,9,17,26}`;
- four CPU experts per layer, 16 total;
- physical NPU expert count per enabled layer: `60`;
- placement SHA256:
  `f6d4e9c6a2e5e8060e846dbc7c628d069c9aa6150aaa1e2690ea28a51ba286a3`;
- F32 GGUF size: `8,858,371,392` bytes;
- F32 GGUF SHA256:
  `2c4cd307a53b761c4191cf108d1e00ec2bd2c99b48a595a28ac8fdd0e0edd1fa`.

## Gates that passed

The frozen pairwise contract did not overflow on either predeclared subset:

| Subset | Positions | Stable exact | Ambiguous membership | Max distortion | Overflow |
|---|---:|---:|---:|---:|---:|
| Q2 | 256 | 137/137 | 119/119 | 2.125 | 0 |
| H2 | 256 | 155/155 | 101/101 | 1.875 | 0 |

All values were finite. Route coverage also passed: 4/4 CPU-enabled layers,
16/16 selected CPU experts, and 74,784 CPU route hits.

## Blocking exact gate

P2 failed N1 same-path determinism in a single process. Ten 64-token greedy
runs produced:

| Prompt | Unique output hashes | First differing token |
|---|---:|---:|
| `v_en_01` | 6 | 1 |
| `v_zh_01` | 1 | none |
| `v_struct_01` | 9 | 3 |
| `v_struct_02` | 9 | 2 |

The 8/16/32-to-64 prefix gate also failed. A sequential control with
`SGLANG_KT_HYBRID_NO_CPU_STREAM=1` remained nondeterministic: `v_en_01`
produced four hashes and `v_struct_01` produced eight. The failure is therefore
confirmed in both overlapped and sequential execution and is not explained by
cross-backend pairwise tolerance.

Failure classification: `SAME_PATH_NONDETERMINISM`.

Expected: one exact token sequence per prompt and exact prefixes.  
Actual: multiple greedy trajectories from identical prompt/configuration in the
same process.  
First failing prompt/token: `v_en_01`, token index 1.  
Next minimal experiment: capture the first differing Layer 1 routed contribution
and scheduler/callback ordering across two sequential-control runs, then localize
the first byte-level divergence before reopening P2.

Per the ordered gate, P2 quality smoke was not run and P3 was not entered.

`MULTI_LAYER_P2 = BLOCKED`
