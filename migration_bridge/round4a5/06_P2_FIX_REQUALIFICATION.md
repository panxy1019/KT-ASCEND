# Round 4A.5：P2 修复后复验

状态：`A3_VERIFIED_READY`

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
| CPU coverage | 4/4 layers、16/16 experts、74,807 CPU route hits，routing weights finite | `evidence/final-p2/route-coverage.json` |
| CPU-not-hit control | 32/1,154 eligible real Layer-17 rows exact | `evidence/final-p2/cpu-not-hit-l17.json` |
| Downstream quality | C-Eval 27/128 vs 27/128；GSM8K 31/128 vs 31/128；paired bootstrap 95% CI 均 `[0,0]` | `evidence/final-p2/p2-quality-paired-summary.json` |
| Retained regression | Round2A 21/21；Round2B 23 pass + 1 expected skip；Round2C 44/44；SGLang KT EP 4/4；Round3/P1 real-model compact PASS | `10_P2_FINAL_REGRESSION.md` |

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

## Quality provenance closure

The exact frozen 128+128 input was recovered in repository history at
`migration_bridge/round4a4/corpora/d_manifest.json`.  Re-serialization with
sorted keys and compact separators reproduces the frozen canonical SHA256
exactly:

`c7bcb09c72f4a7213d0ecdb080f8e88e983ab1a491d4304f448bce28d8f7e1ce`.

The P2 A/B was therefore run against the original frozen rows, not a rebuilt
or selectively sampled replacement.  See `09_QUALITY_PROVENANCE.md` and
`11_P2_FINAL_QUALITY.md` for the recovery chain and the explicit Transformers
environment boundary.

All required P2 gates are closed.  The service was stopped after capture;
port 31000 was released and `npu-smi` reported no running processes on NPU0–7.
