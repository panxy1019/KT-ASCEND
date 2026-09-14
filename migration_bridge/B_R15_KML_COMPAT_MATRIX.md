# B Round 1.5 — Historical KML Compatibility Matrix

## Overall rating

**MEDIUM**

This rating deliberately separates two different historical implementations:

- `c65febe.../kml_kernel` has high source-level proximity to the current generic MoE mat-kernel interface;
- `1a925769.../operators/kml` has low proximity because it is monolithic, directly KML-dependent, exposes old class names, and predates substantial wrapper/config refactoring.

Neither qualifies for an unmodified restoration.

## Contract matrix

| Contract | Historical state | Current `c40d37c...` | Compatibility | Required work if restored |
|---|---|---|---|---|
| `MOEConfig` / `GeneralMOEConfig` | Present in both histories; c65 uses the same generic config family | Present, with additional fields and bindings | Medium-high for c65; low-medium for 1a | Audit initialization of every pointer/type/shape field; do not rely on aggregate layout |
| `CPUInfer` constructor | thread-count and `WorkerPoolConfig` forms existed | `WorkerPoolConfig` path exists and is A3-tested | High at Python surface | Use current constructor and backend pointer; add lifecycle tests |
| WorkerPool | subpool count, NUMA map, thread count existed at c65 | same core fields; header has four added lines since c65 | High source-level | Test current NUMA/error behavior; do not assume privileged memory binding |
| expert mapping | `physical_to_logical_map` and TP mapping used | mapping still exists amid broader expert changes | Medium | Verify logical/physical expert semantics and pointer lifetime |
| weight pointers | gate/up/down plus scales/zeros in generic config | same families, with later config growth | Medium-high | Validate type tags, alignment, packing ownership, and current loader lifetime |
| weight format | old KML path packs int8/int4 and uses scales; c65 kernel is int8/int4 | current generic backend supports multiple configs/wrappers | Medium | Write format-specific golden fixtures; historical packed files are not assumed ABI-stable |
| `MOE_INT8` wrapper | 1a exported `KMLInt8_MOE`; c65 used generic `Int8_KERNEL_MOE` direction | current Python `moe_kernel.py` expects current generic names | Low for 1a; high-ish for c65 | Never reintroduce old pybind names as the primary API; adapt beneath current wrapper |
| pybind class names | `KMLInt8_MOE` / `KMLInt4_MOE` in 1a; newer generic names around c65 | generic `MOE`; optional `Int8_KERNEL_MOE` when built | Low/medium | Bind only the minimal current class set and test symbol presence/absence |
| task interface | load/forward/warm-up tasks through `TP_MOE` existed | submit/sync task surface remains and is tested | Medium-high | Recompile against current templates; test task ownership and exceptions |
| decode API | old custom batch GEMM; c65 implements parent declarations | current `batch_gemm_api.hpp` is byte-identical to c65 | High for c65 | Restore only implementations needed by current declarations; add decode numerical tests |
| prefill API | bundled C int8/int4 GEMM with its own CMake | sources absent; current surrounding build evolved | Medium | Rework CMake target boundaries and ISA probing; avoid fixed SVE assumptions |
| NUMA API | direct NUMA placement and per-node TP behavior | current WorkerPool remains, but container mempolicy was denied | Medium | Test on a permitted but still isolated setup before claiming NUMA correctness |
| buffer ABI | raw pointers, packed buffers, quant scales | raw-pointer model remains; config and wrapper code evolved | Low-medium | Treat as source contract, never binary ABI; rebuild and use sanitizers/golden tests |
| external runtime | 1a includes `kblas.h`; c65 source is mostly custom but KML build switch can link `kml_rt` | KML source absent; config still references SDK/runtime | Low for 1a; unknown for c65 packaging | Pin an official SDK, headers and libraries in a disposable image before any restoration |

## Diff evidence

Between `c65febe...` and current `c40d37c...`:

- `operators/moe_kernel/mat_kernel/batch_gemm_api.hpp`: byte-identical;
- `operators/moe_kernel/la/kernel.hpp`: byte-identical;
- `cpu_backend/cpuinfer.h`: `+8/-4` lines;
- `cpu_backend/worker_pool.h`: `+4/-0` lines;
- `ext_bindings.cpp`: `+396/-21` lines;
- `operators/common.hpp`: `+97/-11` lines;
- `operators/moe-tp.hpp`: `+50/-9` lines;
- `python/utils/moe_kernel.py`: `+11/-5` lines.

The byte-identical mat-kernel API is meaningful evidence that a **minimal, source-level** restoration of the c65 implementation might be tractable. It is not evidence of binary compatibility, current correctness, or official KML 2.5.0 compatibility.

The older `1a925769...` line diverges much more: its monolithic `KML_MOE_TP`, direct `kblas.h` dependency, old pybind names, embedded MLA implementation, wrapper layout, and setup logic do not match the present architecture closely enough for wholesale reuse.

## Restoration assessment

If KML optimization is revisited later, the only defensible starting point is a controlled extraction of the smallest required implementation from `c65febe...`, compiled beneath the current unchanged parent API with new tests. Do not restore both historical trees, do not cherry-pick either deletion's inverse, and do not revive the old `KMLInt8_MOE` public surface.

Required gates before merge would include:

1. pinned, reproducible SDK/runtime acquisition inside a disposable ARM build image;
2. compile-time ISA detection rather than unconditional fixed SVE width;
3. decode and prefill numerical golden tests for int8/int4;
4. current `MOEConfig`, expert-map, task-lifetime and buffer ownership tests;
5. sanitizer/static analysis where supported;
6. KML-off regression proving the default Route A build is unaffected.

Because LLAMAFILE already passes, this restoration work is optional optimization, not a Round 1.5 blocker.
