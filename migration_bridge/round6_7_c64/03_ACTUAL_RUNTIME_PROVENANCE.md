# C64 development provenance freeze

Start: 2026-09-17. Local branch `codex/round6-7-c64-numa`, clean base
`902de84cc31ca7d63a21bcb87641f80e87967373`. No push before operators and canary stabilize.

Actual deployed parent: `e3084d309d6f39b8a9f7185d36ac27e4b77526d7`.
Actual deployed SGLang: `eaa58ccea4ee372b8b207dd16f398eafe3da37ff` plus frozen overlays.
Local gitlinks are not used as a substitute for these deployed revisions.

New container `kt-r67-c64-development-v1`:
`29bef76636f39c2dfbe836d6aaaa401311f5999ac4db66b737bbe9df7965d391`.
NPU2 physical / logical0; CPU128–191, memory nodes4,5;192GiB memory limit;
swap forbidden; SYS_NICE; network none. Original assets and source read-only.
Only `/workspace/c64-dev` is writable evidence/build storage.
R16-C32 and R8-C32 continue untouched on CPU0–31 and64–95 respectively.

Original native SHA `0f3aac9009931b18853c9b8fd6f6bba8b02d6ea396e79b2029bf9ce092049895`.
R8 sparse-NUMA C32 binary is separate and will not receive C64 updates.
Compiler GCC11.4.0, `-O3 -ffast-math -std=gnu++20 -fPIC -fvisibility=hidden
-march=armv8.2-a+fp16+dotprod -fopenmp`. Machine-readable actual paths/hashes and
compiler flags are frozen in target `/workspace/c64-dev/provenance.json`.

Plan distinction: C64 is a new arithmetic profile. Preserve C32 BF16-visible
semantics; C64 FP32 down partials must not be rounded before TP merge. Retain
gate/up/mul and route-weight boundaries unless a separately justified change is
required. F32 tail support must not weaken quantized block alignment.

Performance isolation cannot be achieved while current product tests run.
Do not stop them: complete development/correctness first, then wait for natural
completion before formal isolated A/B. Contended timings are diagnostic only.

Pinned upstream reference inspected:
[LLAMAFILE MoE source at 0b954fe4](https://github.com/kvcache-ai/ktransformers/blob/0b954fe4ca2f9e79519da66156e6a3f92b686232/kt-kernel/operators/llamafile/moe.hpp).
Its FP32 TP output and final conversion are a design reference, not evidence
that our 1408 tail or numerical contract already passes.
