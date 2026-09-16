# Round6.5 provenance — execution in progress

User approved review-v1 on2026-09-16. Parent base c74931235c60d2e899e3acd9fc9277b06962920a;
Round6.4 evidence c14053c5090a5d022cd8580ef7bbd0b08f557f43.
Round5 tag round5-upstream-aligned-tp1-a3-verified dereferences to
be5b48621db32fb894402e3e886060dea263cdc9. No tag changed.

Local branch codex/round6-5-official-style-offload-alignment. Existing llama.cpp
user modifications are untouched. Production source is not rebased or replaced.

Target110.129.0.22, Ascend910B3 NPU0. Container-only runtime snapshot from idle
kt-r64-910b-migration: image29fb2d063e02f8bc0da28e9264c931358f48d040afb4f8b99be2e4239bf825e0.
New container kt-r65-static-offload:
296aef761480b2dbb4cabe1c8ed32c6df2e8926d8af3d85ed972c14abe039c44.
cpuset0–31; memory192GiB; memory+swap192GiB; shm16GiB; NPU0 device only.
Host driver/firmware, model and actual source bind mounts read-only.
Results use existing project artifact volume with new exclusive r65-* roots.

First container inherited /bin/bash entrypoint and exited126 before any task
(`/usr/bin/sleep: cannot execute binary file`). Preserved as
kt-r65-entrypoint-failed-v1, ID e028976d7909971eae38f34799aec0b39f7a00e1a65e933f3d4bbc3ed1d080ad.
New container uses explicit /bin/sleep entrypoint. This was packaging invocation,
not a CPU kernel or model error. No unrelated container changed.

Actual deployed source retains Round6.4 recorded parent/child plus overlays;
new runtime preflight writes actual SHAs/status to r65-preflight-v1/preflight.json.
Local branch HEAD is tooling provenance, not a claim of clean target source.
Tools deployed under /workspace/r65-tools, frozen profiles under
/workspace/results/r65-profiles-v1. All-layer new GGUF under r65-assets-v1;
old D3 weights never overwritten.

R64 compact determinism launcher is being validated while CPU-only export runs.
Its timing and container memory therefore are NOT the controlled capacity or
performance result. Official resource/performance sweep runs after export ends.

Qualification: IN_PROGRESS. Round6.4 IMPLEMENTED_NO_SPEEDUP unchanged.
