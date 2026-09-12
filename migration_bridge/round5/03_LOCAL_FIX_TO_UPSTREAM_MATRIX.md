# Round 5 — Local Fix to Upstream Semantic Matrix

Evidence state: `CODE_INSPECTED`. Final actions are provisional until raw
upstream A3 probes and the Alignment ADR.

| Local fix | Local location | Upstream counterpart | Classification | Provisional final action | Evidence / required probe |
|---|---|---|---|---|---|
| Kunpeng LLAMAFILE CPU expert | `kt-kernel/operators/llamafile/moe.hpp` | same backend plus ARM feature detection | `UPSTREAM_SUPERSET` | `DROP_AS_UPSTREAMED` if R2A passes | build + 1408 shape + 1000 forward |
| F32 GGUF expert weights / BF16 I/O | LLAMAFILE wrapper and loader | GGUF/LLAMAFILE path exists | `UNKNOWN_NEEDS_PROBE` | decide after DSV2 capability probe | exact GGUF load and expert identity |
| non-SVE BF16-weight SGEMM workaround | local LLAMAFILE path | `kt_effective_vec_dot_type` | `UPSTREAM_EQUIVALENT` | `DROP_AS_UPSTREAMED` | A3 CPU probe |
| safe single-threadpool forward | CPUInfer/worker pool | upstream WorkerPool | `UPSTREAM_EQUIVALENT` | `DROP_AS_UPSTREAMED` if lifecycle passes | R2A 1000-forward |
| public CANN vendor bridge | local vendor abstraction | Ascend vendor shim | `UPSTREAM_SUPERSET` | use upstream | U1–U5 probes |
| host callback exception/RAII state | local `cpuinfer.h` | upstream callback state | `LOCAL_SUPERSET` | review for minimal safety port | callback failure/leak tests |
| ACL report processing worker | not first-class locally | upstream callback worker | `UPSTREAM_SUPERSET` | use upstream | subscribe/process/shutdown probe |
| global logical expert IDs | SGLang KTEP | upstream KTEP mapping | `UPSTREAM_EQUIVALENT` | use upstream | C0–C4 mapping tests |
| compact physical NPU weights | local mapping/loader | upstream compact mapping | `UPSTREAM_EQUIVALENT` | use upstream if DSV2 loader passes | physical dim and load audit |
| `int64` router to `int32` NPU boundary | local child | upstream `int32` mapping/remap | `UPSTREAM_EQUIVALENT` | use upstream | dtype assertions |
| valid placeholder ID 0 + zero weight | local child | masking/remap path differs and can use `-1` | `CONFLICTING_SEMANTICS` | probe exact NPU kernel contract | C0–C4 and invalid-index test |
| routing weight exactly once | KTEP/model boundary | same `routed_scaling_factor=None` pattern | `UPSTREAM_EQUIVALENT` | use upstream | explicit sum reference |
| shared expert exactly once | DSV2 model boundary | DSV2 model boundary | `UPSTREAM_EQUIVALENT` | use upstream | Round2C/Round3 |
| D2H/H2D framework `copy_` | local `experts_base.py` | pinned framework copies | `UPSTREAM_SUPERSET` | use upstream | 1000 repeat + P2 determinism |
| blocking visibility before CPU/H2D | local NPU special path | `_wait_device` + host drain | `UPSTREAM_SUPERSET` | use upstream | U2–U4, P2/P3 |
| buffer-lifetime clone after H2D | local SGLang child | pinned parent + upstream child synchronization | `UNKNOWN_NEEDS_PROBE` | retain only if runtime shows shared-buffer hazard | multi-layer capture |
| BF16-visible gate boundary | local `bf16_numerical_compat` | absent | `LOCAL_ONLY_REQUIRED` | minimal port | captured stagewise + pairwise |
| BF16-visible up boundary | local `bf16_numerical_compat` | absent | `LOCAL_ONLY_REQUIRED` | minimal port | captured stagewise + pairwise |
| BF16-visible SwiGLU multiply | local `bf16_numerical_compat` | absent | `LOCAL_ONLY_REQUIRED` | minimal port | captured stagewise + pairwise |
| BF16-visible down boundary | local `bf16_numerical_compat` | absent | `LOCAL_ONLY_REQUIRED` | minimal port | captured stagewise + pairwise |
| BF16-visible route weight | local `bf16_numerical_compat` | absent | `LOCAL_ONLY_REQUIRED` | minimal port | captured stagewise + pairwise |
| FP32 CPU/NPU partial merge + final cast | local SGLang unquant/KTEP | conditional hook exists but no raw DSV2 setter found | `LOCAL_ONLY_REQUIRED` | minimal child port unless raw probe disproves gap | P1 stagewise/pairwise |
| sparse NPU expert load / mmap copy | local loader changes | newer upstream loader architecture | `UNKNOWN_NEEDS_PROBE` | do not port before capability result | DSV2 NPU load |
| CUDA capability guard on NPU | local SGLang child | newer device abstraction | `UPSTREAM_SUPERSET` candidate | use upstream if import/serving passes | KTEP import + all-NPU |
| `SGLANG_APPLY_CONFIG_BACKUP=none` behavior | local operational contract | current SGLang config path | `UNKNOWN_NEEDS_PROBE` | retain only if required | launch probe |
| transfer trace/private scratch/debug knobs | debug-only code | newer telemetry exists | `TEST_ONLY` | port tests, not production complexity | regression inventory |
| OpenBLAS candidate backend | Round4A.2 experiment | no need for alignment | `OBSOLETE` | do not port | backend investigation exhausted |
| Graph/Deferred/Dynamic/MTP/quantized paths | not qualified | upstream implementations | `OUT_OF_SCOPE` | leave off | no READY claim |

## Static decision boundary

The architecture points toward `UPSTREAM_FIRST`, but the ADR remains open until
raw upstream has produced evidence for build, runtime primitives, and the
DeepSeek-V2-Lite BF16/F32 capability profile. In particular, the two numerical
rows marked `LOCAL_ONLY_REQUIRED` cannot be retired on architecture maturity
alone.

