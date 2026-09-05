# Round 4A.5：P2 修复后复验

状态：`P2_EXACT_AND_NUMERICAL_REQUALIFIED__QUALITY_INPUT_MISSING`

本轮使用冻结的 P2 placement（layers `{1,9,17,26}`、16 个 CPU experts）、原始
F32 GGUF、NPU0/TP1/BF16 和 16 CPUInfer workers。生产修复为
`experts_base.py` 中以框架管理的同步 `copy_` 替换 raw ACL host-pointer
D2H/H2D；没有改变 placement 或 CPU MoE arithmetic。

## 已通过

| Gate | Result | Persistent A3 artifact |
|---|---|---|
| Runtime pipeline | 2/2 passed | A3 container runtime test |
| P2 same-process | 9 prompts × 10 × 64、8/16/32 prefix 均 exact | `p2-fix-acceptance-a/same-process-validation9.json` |
| Clean restart | 9/9 exact，且和上一进程逐 prompt exact | `p2-fix-acceptance-b/clean-restart-validation9.json` / `clean-restart-compare-validation9.json` |
| Q2 frozen pairwise | 256 positions，finite，0 overflow，`137/137` stable exact、`119/119` ambiguity membership | `p2-fix-hybrid-requal-v2/q2-contract.json` |
| H2 frozen pairwise | 256 positions，finite，0 overflow，`155/155` stable exact、`101/101` ambiguity membership | `p2-fix-hybrid-requal-v2/h2-contract.json` |
| CPU coverage | 4/4 layers、16/16 experts、74,789 CPU route hits，routing weights finite | `p2-fix-hybrid-requal-v2/route-coverage.json` |
| CPU-not-hit control | 32/1,153 unique real Layer-17 rows exact | `p2-fix-hybrid-requal-v2/cpu-not-hit-l17.json` |

The Q2/H2 validation uses the frozen
`PAIRWISE_NUMERICAL_CONTRACT_CANDIDATE.json` (`B_pair=2.1875`) rather than a
post-fix-fitted bound.  Q2 maximum pairwise distortion was `2.125`; H2 was
`0.75`.

## Review hashes

| Artifact | Canonical SHA256 |
|---|---|
| Q2 metrics | `3865313bfea95b6c903943a47b90182603d2096a59343df2f6b0790004d91aa1` |
| Q2 contract | `c9f7b98f0740bb20cbaad4955e19a675ae6e672e1559a471e2e4c0517fb8fcbd` |
| H2 metrics | `48b814803d6818b04821ff8f753d2c10a96d14a0568c42ae3efc6ae7cb9fb212` |
| H2 contract | `44cbac8a4f190c459f7546d26c6ebac247f33fbd74d4735ce9b6741861381d44` |
| Route coverage | `b055da2b7888af8c8faf181c37cf15c8d3acb1e80a0b794d913183f471d63c2b` |
| CPU-not-hit | `696a9d1cb4c9dab886a9f0dbcc513bd23519b117490c79b1617a126127e37e33` |

## Remaining gate

The frozen 128+128 downstream-quality input that has SHA256
`c7bcb09c72f4a7213d0ecdb080f8e88e983ab1a491d4304f448bce28d8f7e1ce`
is not present in the repository or persisted A3 artifacts: only its manifest
hash and prior All-NPU/P1 result files remain.  Recreating it from a mutable
dataset would not be a valid requalification.  Therefore P2 must not yet be
labelled fully qualified, and P3 remains blocked until that exact frozen input
is restored and the P2 quality A/B is run.

The P2 server was stopped after capture; NPU0 and port 31000 were released.
