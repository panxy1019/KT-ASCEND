# Round 4A.5 P2 Final Retained Regression

Evidence state: `A3_VERIFIED`

This regression was executed after the framework-managed Ascend transfer fix,
after obsolete diagnostic production hooks were removed, and against the P2
closure candidate lineage.  It is a fresh run; the results below do not merely
quote Round 2/3/4A historical reports.

## Execution environment

| Item | Frozen value |
|---|---|
| A3 container | `kt-r4a5-p2-closure-env` |
| container ID | `0825d8eb90afcba292433acc9230780c575f86a5853b34db01d1141ee36ff209` |
| environment image | `sha256:7781588ef508d2cf37bb4a09878b0079bf13795fedd874cd7c76c4da47f2ea3a` |
| CPU / affinity | Kunpeng 920; container cpuset `0-15` |
| accelerator | NPU0 only |
| Python | `3.11.15` |
| torch / torch_npu | `2.9.0+cpu` / `2.9.0.post2` |
| Transformers / tokenizers | `5.3.0.dev0` / `0.22.2` |
| SGLang child | `cbaa79c2f5b004ab4ca470c9fa56161666ccafb2` |
| production transfer source SHA | `ef2c3c136ea40fedbb45e6590c37f52e671cfd517fb3815398eff5d8ce07f571` |
| built `kt_kernel_ext` SHA | `e52eb8361e2f3617b6b1b2df75d6c5066dc61071a94d318c30a77d32a00cfaaf` |

The source was mounted from the isolated Round 4A.5 closure worktree.  CANN's
existing Python path was retained when running NPU tests.  The host, business
containers, system libraries, and NPU1–7 were not modified.

## Results

| Retained suite | Current result | Gate | Compact evidence |
|---|---:|---:|---|
| Round 2A CPU expert plane | 21 passed | PASS | `evidence/final-p2/regression-round2a.log` |
| Round 2B Ascend runtime plane | 23 passed, 1 expected CUDA-only skip | PASS | `evidence/final-p2/regression-round2b.log` |
| Round 2C synthetic Hybrid MoE | 44 passed, 1 non-failing format warning | PASS | `evidence/final-p2/regression-round2c.log` |
| current SGLang KT EP Ascend E2E | 4 passed | PASS | `evidence/final-p2/regression-sglang-kt-ep.log` |
| Round 3 real Layer-17 compact | real CPU/NPU Hybrid checks pass | PASS | `evidence/final-p2/regression-round3-real-layer17.json` |
| Round 4A.4 P1 compact | C0–C4 + 1,000 overlap forwards pass | PASS | `evidence/final-p2/regression-round4a4-p1-compact.json` |

## Round 2A: current FP32-partial contract

The first fresh run reproduced the four known stale expectations from Round
4A.4: tests expected a BF16 CPU wrapper result while the preserved Round 4A.1
Fix B intentionally returns an FP32 routed partial.  The numerical results were
already within the frozen bounds, with observed relative-L2 values below
`0.0047`.

The test-only assertions were corrected to require FP32 at the CPU-expert
partial boundary and to compare against the FP32-materialized reference.  No
production arithmetic or visible-boundary cast was changed.  The corrected
A3 run passed all 21 tests.  Its 1,000-forward CPU lifecycle reported:

```text
iterations = 1000
rss_delta  = 0 bytes
```

This correction is commit `d36dd8f3e01c9a86d87309e972b11e75b537b68e`.

## Round 2B: transfer and lifecycle regression

The suite exercised the Ascend stream handle, 10,000 host callbacks, callback
ordering, D2H/H2D, real CPUInfer, stream lifecycle, and the corrected FP32
output buffer.  The pipeline numerical metrics were:

| Metric | Value | Gate |
|---|---:|---:|
| max absolute error | `0.000244140625` | `<= 1e-3` |
| mean absolute error | `4.5139343e-05` | `<= 1e-4` |
| relative L2 | `0.0049403305` | `<= 1e-2` |

The 1,000-cycle transfer/CPUInfer pipeline reported RSS delta `5,242,880`
bytes, below the 16 MiB test gate.  No non-finite result, callback exception,
deadlock, or timeout occurred.

## Round 2C: Hybrid coordinator regression

All 44 tests passed.  Coverage includes fixed placement, logical/physical
mapping, CPU-only/NPU-only/mixed routing, sequential and overlapped Hybrid,
repeatability, 1,000 mixed cycles, 20 wrapper recreations, and 100 stream
create/destroy cycles.

An initial invocation hid `acl.so` by replacing rather than extending
`PYTHONPATH`; eight overlap/lifecycle tests therefore stopped at `import acl`.
No affected test reached Hybrid execution.  Re-running with the staging path
prepended to the original CANN path passed all 44 tests.  This is classified as
an invocation/environment error, not a code regression and not a passing run.

## Round 3 real-model compact result

The Layer-17 verifier used the frozen DeepSeek-V2-Lite snapshot and real F32
GGUF, with expert 8 on CPU and the remaining experts on NPU.  Results:

| Check | Result |
|---|---:|
| CPU-not-hit vs All-NPU | exact, rel-L2 `0` |
| KTEP vs explicit `CPU + NPU` | exact, rel-L2 `0` |
| mixed vs All-NPU | rel-L2 `0.0032707124` |
| final shared/scaled vs All-NPU | rel-L2 `0.0031241263` |
| finite routed/final | PASS |

## Round 4A.4 P1 compact result

The frozen P1 placement remained Layer 17 / CPU experts `{6,8,25,36}` / 60
physical NPU experts.  C0–C4 all matched an explicit CPU+NPU reference exactly.
The largest Hybrid-vs-All-NPU expert relative-L2 was `0.0035398305`, below
`1e-2`.  All 1,000 overlapped forwards produced exactly one routed hash:

`878e638f32c3311f816f3dc01f6e83dcd822c4f332083a591675d3ccaaafaba2`.

The final shared/scaled hash was:

`2f25017046177a407220bae1cf240e8451e57bda34a4461347b4615329b4db17`.

## Transfer-path behavioral protection

The P2 same-process serving qualification is the primary behavioral regression
for the actual fault: 90 64-token requests on a four-layer placement must yield
one exact trajectory per prompt, and all 8/16/32 prefixes must agree.  This
test failed under raw pointer transfers and passes under framework-managed
copies.  Round 2B/2C additionally cover low-level transfer and overlap
lifecycle behavior.  A source grep alone is not treated as the regression.

## Artifact file hashes

| Artifact | File SHA256 |
|---|---|
| Round 2A log | `6e436d4f71f196d9db7a78a609d6336eba2f1ce926fff8bb65c4512c47896b23` |
| Round 2B log | `c8676f0271e701588d96d903d481e97697d8a799f49213a10a5b1a00cb864466` |
| Round 2C log | `cd7b143b3e0b88cac9c49066ee785272f845623dc482c476053e6d42d678ad9f` |
| SGLang KT EP log | `581c993d0129155c0b8587fc393bd538bd26ca09dbe1c3b14c8feccf27f5b122` |
| Round 3 result | `da508c7e52846518aa7b4fbddd978e21da3e7e54b31d2bb9e3acfa3a3e91108a` |
| P1 compact result | `0ae7e293be538de18e69f7410f8210daac3d923a2de4ae0dd9374732ed398f14` |

## Decision

```text
P2_RETAINED_REGRESSION = PASS
LIFECYCLE_STABILITY    = PASS
```

No unexplained retained-suite failure remains.
