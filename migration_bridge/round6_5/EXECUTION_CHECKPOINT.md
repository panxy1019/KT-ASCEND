# Round6.5 execution checkpoint

2026-09-16: User approved review-v1 and autonomous in-scope execution.

- Branch codex/round6-5-official-style-offload-alignment, base c74931235c60d2e899e3acd9fc9277b06962920a.
- WP0/WP1 capacity and full-weight identity complete; compact R64 runtime PASS (28requests including independent restart).
- Live host preflight 11:32 UTC: ~987GiB available, no NPU processes, 256 physical cores, eight NUMA nodes of32cores each. NUMA0=0–31; NPU0 PCI c1:00.0 reports NUMA6.
- Keep32workers: expanding beyond32 crosses NUMA. Preserve historical affinity for controlled ratio comparison; report remote NPU locality, do not claim optimal topology.
- New dedicated container192GiB cap; old project idle container stopped after namespace-conflict attribution, data untouched. No other workloads modified.
- New all-layer GGUF required; existing main GGUF only historical D3 selected layers. Model config confirms26MoE layers,64experts,TopK6,H2048,I1408.
- Original llama.cpp user modification preserved.

Active bounded queue: /workspace/results/r65-single-entry-v1.
Waits for r65-r64-det-v3 then R64 numerical capture → singleR32 determinism →
same-history numerical capture → frozen pairwise evaluation. Stops on any failure;
does not enter four/all-layer phase before outstanding local controls close.

Target container kt-r65-static-offload. Full asset r65-assets-v1/all-moe-f32.gguf
SHA20114dba7790a9b6be40a17ffb3881c516b81ea86b8f2fcfd3ea61d052e96fe2;
identity.json4992/4992PASS. R64v1/v2environment failures preserved, repaired
respectively namespace ownership and discarded CANN PYTHONPATH. See00provenance.

Final qualification NOT_COMPLETE; Round7 NOT_STARTED; Deferred OFF.

12:04 UTC update: R64 numeric capture completed with512target positions and8
lookahead forwards, integrity PASS. SingleR32 same-process24requests exact;
independent restart running, followed by same-history numeric comparison.
Local real-weight safety queue r65-single-controls-v1 waits for successful
singleR32 pairwise result, then E1/T1,T4 ten repeats and controlled NPU-owned
route zero output. Synthetic hidden scope is explicit, not captured-input proof.

Resource monitor v1 started during singleR32 repeat (not before model loading);
that trace is partial diagnostics, NOT a full HBM loading curve. Future full
sampling includes PSS/cgroup in addition to anon/file/faults/NUMA. No CPU count
expansion or production source changes performed.
