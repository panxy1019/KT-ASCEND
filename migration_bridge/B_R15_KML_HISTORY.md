# B Round 1.5 — Public KML Source Archaeology

## Method and safety

The two historical snapshots were inspected in temporary Git worktrees. No historical source was copied into the current branch, no cherry-pick was performed, and no historical binary was built or installed.

| Lineage | Source snapshot | Deletion boundary |
|---|---|---|
| monolithic `operators/kml` | `1a925769d9b3a71d1755ceea27a73ab5c2ee18ea` | `f854d03bd780f10b03f3338b38a6a54d21b7ec1c` |
| integrated `kml_kernel` | `c65febe05ca26f829f85a35e6387cf210d4c649f` | `53f6a6d6e1327be701dfe7318cc6e18cd5b51b6a` / PR #1704 |

## Lineage 1: `operators/kml`

The `1a925769...` tree contains 51 files grouped as follows:

| Area | Contents / role |
|---|---|
| `la/` | `arm_kml.hpp`, batch-GEMM API, decode kernels, utilities |
| `mla/` | DeepSeek V3, gate, MLA and quantized decode/prefill implementations |
| `moe.hpp` | `KML_MOE_TP`, weight packing/loading, decode and prefill routed experts |
| `prefillgemm/` | int8 C kernels, packing, driver, interface, post-ops, CMake and test |
| `prefillgemm_int4/` | int4 equivalent, CMake and test |
| `test/` | conversion, int4 multiplication, matrix and utility tests |

Key contracts:

- `moe.hpp` directly includes `<kblas.h>`, so the monolithic backend has an external KML header/runtime dependency.
- It plugs into the then-current `TP_MOE`/`GeneralMOEConfig` abstraction.
- It has explicit `forward_decode` and `forward_prefill` paths.
- Decode calls custom int8/int4 batch-GEMM entry points; prefill calls the bundled integer GEMM C implementations.
- It packs/loads per-expert gate, up, down weights and associated scales.
- Its pybind names were `KMLInt8_MOE` and `KMLInt4_MOE`.

### Deletion `f854d03...`

The commit dated 2025-11-03 has the one-line message `update kt-kernel`. For the inspected paths, it deletes the complete `operators/kml` tree and changes CMake, setup, `ext_bindings.cpp`, and `python/experts.py`. The inspected stat is 53 files, 287 insertions, and 12,213 deletions; the historical KML tree accounts for 12,089 deleted lines.

The pybind change removes `KMLInt8_MOE`/`KMLInt4_MOE` and introduces the newer `Int8_KERNEL_MOE`/`Int4_KERNEL_MOE` bindings behind `USE_MOE_KERNEL`. Setup simultaneously adds/retains ARM auto-enabling logic for `CPUINFER_ENABLE_KML`, which helps explain why later snapshots can still expose a KML build flag after the original implementation disappeared.

Exact removal motivation: **NOT VERIFIED**. The commit message contains no rationale, issue, test evidence, or compatibility statement.

## Lineage 2: `moe_kernel/mat_kernel/kml_kernel`

The `c65febe...` tree contains 33 files:

| Area | Contents / role |
|---|---|
| root | `batch_gemm.cpp`, `batch_gemm_kernels.cpp`, `utils.hpp` |
| `prefillgemm/` | int8 integer GEMM pipeline, packing, kernels, interface, CMake and test |
| `prefillgemm_int4/` | int4 counterpart, CMake and test |

This is not a verbatim copy of the older monolithic backend. It is an implementation beneath the newer generic `MOE_KERNEL_TP` mat-kernel API:

- `batch_gemm.cpp` includes the parent `batch_gemm_api.hpp` that still exists in current mainline.
- decode uses custom SVE int8/int4 microkernels;
- prefill uses the bundled C integer GEMM trees;
- the prefill CMake uses `-march=armv8.3-a+sve+i8mm` and a fixed 32-byte SVE configuration;
- its source does not itself include `kblas.h`; KML runtime linkage remains a build-system concern elsewhere rather than an obvious direct dependency of these 33 files.

### Deletion `53f6a6d...` / PR #1704

The 2025-12-11 merge deletes exactly all 33 files and 5,401 lines from `kml_kernel`. It does **not** complete a repository-wide removal: the parent declarations, benchmark references, and later/current KML configuration references remain.

The public [PR #1704](https://github.com/kvcache-ai/ktransformers/pull/1704) contains only a template-level description and the author comment `remove bugs`. The automated review explicitly warned that deleting the implementation left dangling declarations and a KML benchmark dependency. Thus the verified intent is “remove bugs” by deleting the module; the concrete defect, failing configuration, and why deletion was preferred over repair are **NOT VERIFIED**.

## Deletion-boundary answers

| Question | `f854d03...` | `53f6a6d...` |
|---|---|---|
| Source deleted? | entire monolithic KML/MLA/MoE tree | entire 33-file `kml_kernel` tree |
| CMake changed in same commit? | yes, but KML configuration was not fully retired | no relevant cleanup in the deletion stat |
| setup changed? | yes; later KML auto-enable semantics remain | no |
| Python/pybind changed? | pybind KML class names removed/replaced; `experts.py` changed | no corresponding wrapper/benchmark cleanup |
| Dead flags/references left? | yes | yes; current mainline still has them |
| Exact technical reason documented? | NOT VERIFIED | NOT VERIFIED beyond vague `remove bugs` |

## Current-mainline implication

Round 1's finding is confirmed: current `c40d37c...` has build configuration that can reference `operators/kml`, `kml_kernel`, `kml_rt`, and `CPU_USE_KML` although the referenced source trees are absent. That inconsistency is public upstream history, not damage to the A3 environment.

History proves provenance, but it does not establish that either backend should be restored.
