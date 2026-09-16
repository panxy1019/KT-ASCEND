# Round6.5 execution checkpoint

2026-09-16: User approved review-v1 and autonomous in-scope execution.

- Branch codex/round6-5-official-style-offload-alignment, base c74931235c60d2e899e3acd9fc9277b06962920a.
- WP0/WP1 IN_PROGRESS; no model tests started.
- Live host preflight 11:32 UTC: ~987GiB available, no NPU processes, 256 physical cores, eight NUMA nodes of32cores each. NUMA0=0–31; NPU0 PCI c1:00.0 reports NUMA6.
- Keep32workers: expanding beyond32 crosses NUMA. Preserve historical affinity for controlled ratio comparison; report remote NPU locality, do not claim optimal topology.
- Old container has no memory cap; new dedicated container will have explicit cap. Do not modify old container or other workloads.
- New all-layer GGUF required; existing main GGUF only historical D3 selected layers. Model config confirms26MoE layers,64experts,TopK6,H2048,I1408.
- Original llama.cpp user modification preserved.

Final qualification NOT_COMPLETE; Round7 NOT_STARTED; Deferred OFF.
