# B Round 1.5 Summary and Decision

## CPU-only build

**PASS / A3_VERIFIED**

Current `c40d37c...` builds as an aarch64 CPU-only extension with KML, optional CPU MoE kernel, MLA, and all GPU backends disabled. Ordinary build dependencies were installed only in a disposable container; the A3 host was not modified.

## CPUInfer

**PASS / A3_VERIFIED**

`kt_kernel`, `kt_kernel_ext`, `WorkerPoolConfig`, `CPUInfer`, `MOEConfig`, and generic `MOE` import. A real routed-expert load/forward task completed 1,000 submit/sync iterations without a crash and matched the BF16-output reference exactly for the fixture.

## LLAMAFILE on Kunpeng

**LLAMAFILE_ARM_READY**, qualified to the tested core backend.

The actual aarch64 LLAMAFILE sources compiled and the generic C++ MoE path ran correctly. GGUF ingestion, quantized GGUF weights, NUMA memory binding, model-sized performance, and full-model integration remain unverified and are next-round test gates.

## Historical KML

```text
operators/kml source:  FOUND at 1a925769...; deleted at f854d03...
kml_kernel source:     FOUND at c65febe...; deleted at 53f6a6d... / PR #1704
```

The public history confirms current mainline's dangling KML configuration/source mismatch. Neither deletion contains a sufficiently specific technical rationale; the exact cause is **NOT VERIFIED**.

## Historical-current compatibility

**MEDIUM overall**

The newer c65 `kml_kernel` has high source-level proximity: its parent batch-GEMM API and LA kernel header are byte-identical to current. The older monolithic backend is low compatibility and directly includes `kblas.h`. Current config, bindings, wrappers, ownership and build logic have evolved enough that no historical directory is safe to restore wholesale.

## KML runtime

**AVAILABLE package; A3 OS compatibility UNKNOWN**

Huawei officially exposes KML 2.5.0 and documents `kblas.h`, `kml_rt`, `/usr/local/kml`, and HPCKit module paths. The inspected KML/HPCKit 24.0 support material does not establish support for openEuler 24.03 LTS-SP1. Nothing was installed on A3.

## Decision

**Route A — LLAMAFILE first**

All three Route A gates are satisfied:

```text
CPU-only build             PASS
CPUInfer                   PASS
LLAMAFILE ARM core runtime PASS
```

The first Hybrid MoE MVP should therefore use LLAMAFILE for CPU experts. KML is demoted to an optional optimization investigation. Restoring KML now would add SDK provenance, OS compatibility, fixed-SVE assumptions, raw-buffer ABI, build-system and regression risk without removing a current functional blocker.

## Next blocker / next-round gate

The next legitimate blocker is no longer “can kt-kernel run on Kunpeng without KML?” That question is resolved positively. The next round should validate the LLAMAFILE route at its remaining integration boundary:

1. add a minimal deterministic GGUF expert fixture and exercise `LlamafileMoEWrapper`/`GGUFLoader` end to end;
2. correct ARM CPU-variant detection so metadata does not falsely report `avx2`;
3. add multiple experts/top-k and prefill-shape correctness tests;
4. establish safe NUMA/affinity observability without modifying the host;
5. only after those gates, connect the CPU expert path to the separately designed Hybrid MoE integration.

KML restoration should be a later independent optimization task. If approved, start only from the minimal c65 mat-kernel implementation beneath current APIs, in a disposable image with a pinned and verified official package; do not inverse-cherry-pick either deletion and do not revive both historical trees.

## Explicit non-actions

Round 1.5 added no Ascend vendor header, CANN callback, NPU code, SGLang change, model execution, TP2+, graph, deferred-expert, or dynamic-placement implementation. The only repository addition beyond documentation is the isolated smoke-test asset under `migration_bridge/experiments/`.
