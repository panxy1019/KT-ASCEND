# Round 5 — R2A / R2B / R2C Requalification

Evidence state: `A3_VERIFIED` unless explicitly labelled otherwise.

## R2A — Kunpeng LLAMAFILE CPU expert plane

The final current-interface core matrix passed `18/18`:

- aarch64 detector and extension metadata;
- real GGUF load and routed numerical correctness;
- identity and non-identity physical-to-logical mapping;
- two-layer loader isolation;
- 1000 repeated forwards with stable lifetime.

The earlier retained `21/21` suite contains three assertions against local
private diagnostics removed by upstream (`_gguf_loader_path` and
`cpu_runtime_diagnostics`).  They are classified `STALE_TEST_EXPECTATION`, not
restored as production compatibility APIs.  Upstream's loader cache and worker
configuration remain the authoritative replacement.

```text
R5_R2A_CORE = PASS (18/18)
R5_R2A_STALE_PRIVATE_ASSERTIONS = 3
R5_R2A_UNEXPLAINED_FAILURES = 0
```

Evidence: `a3_aligned_round2a_core_final.log` and
`a3_aligned_round2a_mapping_v2.log` in the A3 Round 5 artifact root.

## R2B — Ascend runtime plane

The historical test files were preserved and run first.  Their raw outcome
was `6 passed, 1 skipped, 14 failed, 3 errors`.  Every failure was traced to an
obsolete test-only interface rather than hidden acceptance relaxation:

- `submit_with_device_stream` / `sync_with_device_stream` were local aliases;
  pinned upstream exposes `*_with_cuda_stream` with a CUDA-shaped vendor ABI;
- `kt_kernel_ext.testing.CPUInferTestTask` was a local test binding absent from
  upstream production builds;
- old callback fixtures compile against the former vendor include layout.

The same behavior was then exercised using the current public interface:

- callback worker init, active-stream subscription, explicit shutdown: PASS;
- BF16, FP32 and int64 pinned D2H/H2D, 1000 repetitions each: exact;
- pageable host tensor fail-fast: PASS;
- real LLAMAFILE D2H → CPUInfer task → H2D pipeline: PASS;
- real pipeline, 1000 cycles: one sampled output, all finite, RSS delta 0.

The committed A3 regression is
`kt-kernel/test/round5/test_ascend_framework_transfer.py`; result `5/5 PASS`.

```text
R5_R2B_FRAMEWORK_TRANSFER = PASS
R5_R2B_REAL_CPUINFER_PIPELINE = PASS
R5_R2B_1000_CYCLE_LIFECYCLE = PASS
R5_R2B_UNEXPLAINED_FAILURES = 0
```

## R2C — Hybrid semantics

Pinned upstream intentionally removed the local standalone synthetic Hybrid
coordinator.  Reintroducing it solely to run the old 44 tests would violate the
Round 5 invariant that KTEP is the single integration point.

Equivalent current-architecture evidence is:

- SGLang KTEP alignment contracts: `4/4 PASS`;
- CPU-owned routes replaced by valid NPU placeholder ID 0 with exact zero
  route weight;
- mapping converted to contiguous int32 at the LLAMAFILE boundary;
- CPUInfer receives the active accelerator stream handle;
- real Layer17/Expert8 strict replay executes both CPU and NPU routes;
- KTEP output equals the explicit `CPU partial + NPU partial` reference
  exactly in the captured case;
- CPU-not-hit merge is exact and mixed CPU/NPU output remains finite;
- same-process ten-repeat real-model trajectory is unique.

```text
R5_R2C_CURRENT_KTEP_SEMANTICS = PASS
R5_R2C_OBSOLETE_COORDINATOR_TESTS = NOT_APPLICABLE_BY_ARCHITECTURE
R5_R2C_UNEXPLAINED_FAILURES = 0
```

This is a semantic requalification, not a claim that deleted private API is
source-compatible.

