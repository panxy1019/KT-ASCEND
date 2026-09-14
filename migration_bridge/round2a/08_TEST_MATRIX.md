# Round 2A Test Matrix

## Final result

Final combined A3 command executed the four focused modules against the freshly built aarch64 CPU-only extension:

```text
test_cpu_detect_metadata.py
test_llamafile_gguf_e2e.py
test_llamafile_routed_correctness.py
test_llamafile_cpuinfer_lifecycle.py
```

Result: **21 passed in 4.82s**.

Breakdown: ARM metadata 6 tests, GGUF E2E 2 tests, routed correctness 8 parametrized tests, lifecycle/NUMA 5 parametrized tests.

## Required matrix

| Test | A3 ARM | Evidence / note |
|---|---:|---|
| ARM variant detection | PASS | actual `aarch64 -> arm`; 6 metadata/unit cases |
| x86 AVX2 regression | PASS | mocked x86 flags continue to select AVX2 |
| CPU-only build | PASS | aarch64 extension built with GPU backends and KML OFF |
| import | PASS | extension import, metadata `arm` |
| GGUF fixture generation | PASS | local-only generator |
| GGUF fixture reproducibility | PASS | two files SHA-256 `e2a275...6247` |
| GGUFLoader | PASS | 6 tensors/2 layers loaded, F32 types |
| KTMoEWrapper dispatch | PASS | E2E uses production wrapper dispatch |
| LlamafileMoEWrapper load | PASS | gate/up/down keys and MOE load task |
| single selected contribution | PASS | weight 1/0 edge verifies one selected contribution |
| 4 experts top2 | PASS | explicit `[1,3]`, weights `[0.7,0.3]` |
| 8 experts top2 | PASS | permutation routed reference |
| physical/logical mapping | PASS | identity plus non-identity permutation |
| qlen=1 | PASS | decode path |
| qlen=2 | PASS | flattened-token/batch-like case |
| qlen=8 | PASS | multi-token routing |
| qlen=32 | PASS | grouped prefill path |
| qlen=64 | PASS | grouped prefill path |
| weight=1.0 / 0.0 | PASS | BF16 numerical thresholds |
| same/different experts across tokens | PASS | edge routing case |
| 1000 repeated forward | PASS | deterministic; RSS delta 0 |
| create/destroy loop | PASS | 20 iterations minimum |
| same/different GGUF path | PASS | canonical-path loader cache |
| two layer wrappers | PASS | layer 0 and layer 1 keys isolated |
| threadpool_count=1 | PASS | `[node 0: 4 threads]` |
| threadpool_count=2 | PASS | `[node 0:2, node 1:2]` |
| NUMA diagnostics | PASS | config, affinity, nodes observable |
| actual NUMA membind | NOT CLAIMED | unprivileged container rejects memory policy calls |
| no KML dependency | PASS | build flag OFF; KML/optimized MOE kernel OFF |
| no NPU use | PASS | no devices; autoload disabled; CPU torch |
| no host modification | PASS | dependencies only inside disposable container |

## Numerical envelope

| Metric | Acceptance | Worst observed | Result |
|---|---:|---:|---:|
| max absolute error | `<= 1e-3` | `6.103515625e-05` | PASS |
| mean absolute error | `<= 1e-4` | `1.4901161193847656e-08` | PASS |
| relative L2 error | `<= 1e-2` | `3.9346272514744805e-05` | PASS |

Configuration: BF16 I/O, F32 weights/GGUF, hidden=256, intermediate=256 (or 512 for two-pool split), experts=4/8, top-k=2, seed=20260827.

## Failure ledger

| Classification | First symptom | Root cause | Resolution | Blocking now? |
|---|---|---|---|---:|
| OTHER | CPU-only wrapper construction raised pinned allocator error | all-CPU mask unnecessarily required pinned allocator | narrow fallback to ordinary CPU mask | No |
| PREFILL | qlen 32/64 non-finite with hidden 32 | grouped LLAMAFILE down path requires 256-aligned hidden blocks | fail-fast validation; minimum fixture hidden=256 | No |
| NUMERICAL | exact-zero assertion failed at ~1e-5 | inappropriate bit-exact assertion for BF16 kernel result | explicit max/mean/relative-L2 thresholds | No |
| BUILD portability | global ARM `-march` is stronger than generic aarch64 | current CMake hard-codes SVE/BF16/dotprod | recorded `P1_BUILD_PORTABILITY_RISK` | No for A3; yes before generic ARM release |

## Evidence location

Remote retained audit root: `/home/admin/kt_round2a_c40d37c/logs/round2a/`. The authoritative aggregate log is `full-test-matrix.log`; focused logs are listed in `00_ROUND2A_ENVIRONMENT.md`.
