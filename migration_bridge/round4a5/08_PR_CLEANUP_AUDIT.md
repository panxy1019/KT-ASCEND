# Round 4A.5 PR Cleanup Audit

Status: `CLEANUP_APPLIED_LOCAL_VERIFIED`

Evidence state: `CODE_INSPECTED`

Takeover head: `305b84359d6a646ab260a16d004d19147103d561`

## Production fix audit

Commit `00c7acfe3569a6a7a6af5b9891e9989e334dffb3` changes the normal Ascend CPU
expert path in `kt-kernel/python/experts_base.py` from raw pointer-based
`acl.rt.memcpy` D2H/H2D to blocking framework-managed tensor `copy_` calls.

The production behavior is now:

```text
NPU hidden / IDs / route weights
        ↓ torch Tensor.copy_(..., non_blocking=False)
pinned CPU tensors retained in _npu_pending_forward
        ↓ CPUInfer task + sync
FP32 CPU routed partial
        ↓ torch Tensor.copy_(..., non_blocking=False)
NPU FP32 routed partial
```

The existing explicit `acl.rt.synchronize_stream()` before D2H remains. The
stream handle is borrowed from torch_npu; CPUInfer neither creates nor destroys
it. No raw `acl.rt.memcpy` remains in the normal `kt-kernel/python` CPU-expert
transfer path.

## Integration invariants

| Invariant | Audit result | Evidence basis |
|---|---|---|
| I1 SGLang remains model/serving source of truth | PASS | frozen child `cbaa79c2...`; `KTEPWrapperMethod` remains in SGLang |
| I2 `KTEPWrapperMethod` remains the integration point | PASS | no replacement wrapper introduced |
| I3 no duplicate Hybrid coordinator | PASS | PR changes only existing wrapper/backend and diagnostic tools |
| I4 global logical expert ID remains canonical | PASS | mapping code unchanged by production fix |
| I5 CPU/NPU ownership remains disjoint | PASS | ownership masks and CPU-owned NPU route disable unchanged |
| I6 same logical expert uses same checkpoint revision | PASS | model, GGUF, placement, and child revisions frozen |
| I7 routing weight applied once | PASS | transfer-only production diff; prior P1/P2 controlled evidence retained |
| I8 routed scaling applied once | PASS | SGLang runner still disables inner scaling and outer model owns scaling |
| I9 shared expert applied once | PASS | outer model ownership unchanged |
| I10 CPUInfer does not own NPU stream | PASS | borrowed handle; no create/destroy in fix |
| I11 callback does not launch/synchronize arbitrary NPU work | PASS | production fix adds tensor copies outside host callback |
| I12 host buffers live through consumer completion | PASS | tensors retained in `_npu_pending_forward` through `cpu_infer.sync()` |
| I13 All-NPU/Hybrid model and tokenizer revisions match | PASS | frozen execution metadata; must be rechecked on A3 closure run |
| I14 advanced CUDA paths remain out of scope | PASS | fix is limited to Ascend CPU expert transfer branch |
| I15 transfer participates in framework dependency/lifetime semantics | PASS | blocking Tensor `copy_` is the normal D2H/H2D path |

These are code-audit results. The final A3 run must still verify behavior at the
cleaned commit.

## Diagnostic cleanup classification

| File | Symbol / environment variable | Purpose | Root-cause relevance | Long-term value | Default changed? | Decision | Reason |
|---|---|---|---|---|---:|---|---|
| `kt-kernel/python/experts_base.py` | `_cpu_tensor_sha256`, `_write_cpu_task_trace`, `_cpu_task_trace_sequence`, `KT_DEBUG_CPU_TASK_TRACE_FILE` | hash CPU task inputs/outputs and append JSONL ordering trace | first-stage attribution support | evidence already preserved; high hot-path complexity if armed | no | `REMOVE_BEFORE_MERGE` | production module should not own file-based forensic tracing after attribution |
| `kt-kernel/python/experts_base.py` | `isolation_key`, `KT_DEBUG_PER_WRAPPER_CPUINFER` | give each wrapper a private CPUInfer queue | negative control rejected shared queue as sole cause | low after attribution | no | `REMOVE_BEFORE_MERGE` | changes cache identity and lifecycle solely for a completed negative experiment |
| `kt-kernel/python/experts_base.py` | `KT_DEBUG_NPU_CPU_INPUT_CLONE` | clone D2H destinations before CPU task | decisive negative control against destination reuse alone | low after attribution | no | `REMOVE_BEFORE_MERGE` | extra allocation/copy branch is no longer required in production code |
| `kt-kernel/python/experts_base.py` | `KT_DEBUG_NPU_CPU_OUTPUT_CLONE` | clone CPU output before H2D | H2D-source ownership negative control | low after attribution | no | `REMOVE_BEFORE_MERGE` | retains extra state and allocation solely for diagnosis |
| `kt-kernel/operators/llamafile/moe.hpp` | `KT_DEBUG_LLAMA_PRIVATE_SCRATCH`, `use_private_scratch_`, private allocator | isolate LLAMAFILE scratch ownership | negative control; nondeterminism remained | low | no | `REMOVE_BEFORE_MERGE` | adds a second allocation/lifetime implementation with no production benefit |
| `kt-kernel/operators/moe-tp.hpp` | `KT_DEBUG_MOE_PRIVATE_OUTPUT`, `use_private_output_` | isolate TP merge output storage | negative control; nondeterminism remained | low | no | `REMOVE_BEFORE_MERGE` | adds production C++ branching for a rejected hypothesis |
| `migration_bridge/round4a4/tools/launch_quality_placement.sh` | `KT_PLACEMENT_PATH`, `KT_GPU_EXPERTS_RATIO` | run diagnostic placement subsets | essential to reproduce layer bisection | useful for future test-only reproduction | no | `KEEP_TEST_ONLY` | script is under migration tooling, default remains the frozen profile |
| `migration_bridge/round4a4/tools/launch_quality_placement.sh` | `KT_CPUINFER_THREADS` | worker-count A/B | negative localization control | useful for test-only reproduction | no | `KEEP_TEST_ONLY` | not part of serving configuration unless explicitly set in the test launcher |
| `migration_bridge/round4a3/tools/run_same_path_determinism.py` | prompt filters, fixed suffix, token-length/prefix switches | minimal deterministic reproduction and matched-history preparation | directly useful for behavioral regression | high | no | `KEEP_TEST_ONLY` | deterministic tool surface; validates arguments and preserves default 64/prefix behavior |
| `migration_bridge/round4a3/tools/run_teacher_forced_sweep.py` | dual history-schema reader | accept current determinism evidence schema | required for post-fix qualification | high | no | `KEEP_TEST_ONLY` | compatibility is isolated to qualification tooling |
| `migration_bridge/round4a5/tools/analyze_kt_numerical_dumps.py` | dump comparator | identify first differing layer/stage | root-cause evidence producer | high for audit/reproduction | no | `KEEP_TEST_ONLY` | standalone deterministic analysis tool |
| `migration_bridge/round4a5/tools/make_diagnostic_placement.py` | diagnostic subset generation | layer bisection without changing frozen acceptance placement | root-cause evidence producer | high for audit/reproduction | no | `KEEP_TEST_ONLY` | output is explicitly diagnostic and does not replace P2 placement |
| `migration_bridge/round4a5/tools/run_llamafile_multilayer_replay.py` | fixed-input direct CPU replay | separate CPU arithmetic from serving lifecycle | important negative control | high for regression investigation | no | `KEEP_TEST_ONLY` | standalone test tool with preserved evidence |
| `migration_bridge/round4a5/evidence/*` | negative and positive experiment artifacts | preserve attribution chain | essential | permanent audit value | no | `KEEP_TEST_ONLY` | negative evidence must not be deleted during cleanup |
| `kt-kernel/python/experts_base.py` | framework-managed blocking `copy_` D2H/H2D | production transfer fix | decisive A/B | required | yes, intentionally | `KEEP_PRODUCTION` | restores torch/torch_npu stream and tensor-lifetime participation |
| `kt-kernel/test/per_commit/test_ascend_runtime_pipeline.py` | FP32 routed-output allocation | correct retained test contract | closes false BF16 buffer overflow/failure | required | test only | `KEEP_TEST_ONLY` | LLAMAFILE exposes FP32 routed partial before final model-boundary cast |

The historical `KT_DEBUG_NPU_TORCH_TRANSFERS` gate is already absent at the
takeover head. Its framework-managed branch was promoted to normal production
behavior by `00c7acfe...`; the unsafe raw-transfer alternative was removed.

## Cleanup boundary

The cleanup commit may remove only the default-off diagnostic hooks classified
`REMOVE_BEFORE_MERGE`. It must preserve:

- the framework-managed production transfer fix;
- the corrected FP32 retained runtime test;
- all diagnostic tools and evidence files;
- the default frozen P1/P2/P3 launcher behavior;
- model, placement, pairwise, and quality contract metadata;
- the original friend history before the cleanup commit.

Cleanup must not alter routing, mapping, ownership, arithmetic, scaling, shared
expert behavior, model/weight loading, or SGLang child revision.

## Decision

```text
KEEP_PRODUCTION              = framework-managed D2H/H2D copy_
KEEP_TEST_ONLY               = tools, corrected tests, and evidence
KEEP_DEFAULT_OFF_DIAGNOSTIC  = none in production source after cleanup
REMOVE_BEFORE_MERGE          = six production diagnostic facilities
PRODUCTION_FIX_AUDIT         = PASS_CODE_INSPECTED
```

## Cleanup execution

The takeover baseline was committed before cleanup as:

```text
b24b72800cd0c8e2b8d25acd6df032dadb1533ea
chore(round4a5): record takeover baseline before cleanup
```

The six production diagnostic facilities classified
`REMOVE_BEFORE_MERGE` were then removed in the isolated commit:

```text
8a9c0d0ae8e6320e234a7097b7e6874b081cf292
refactor(round4a5): remove obsolete diagnostic production hooks
```

Local verification at `8a9c0d0a...` established:

- all six removed diagnostic symbols are absent from `kt-kernel`;
- `moe.hpp` and `moe-tp.hpp` are byte-equivalent to the frozen Round 4A.4
  base, so rejected C++ isolation branches were removed completely;
- the remaining `experts_base.py` delta against Round 4A.4 is the production
  framework-managed transfer fix and its explanatory comments;
- Python byte-compilation and `git diff --check` pass;
- the SGLang child remains unchanged and clean at `cbaa79c2...`.

This is `LOCAL_VERIFIED`, not an A3 qualification result. Final P2 behavioral
tests must run at a descendant of this cleanup commit.
