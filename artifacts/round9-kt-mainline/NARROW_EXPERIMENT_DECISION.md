# Narrow experiment decision — final bounded-scope decision

T5 remains NOT_ENTERED. No scheduling or kernel behavior has changed.

Current lead: expert shadow preparation. The normal5-token sampled request recorded65prepare calls and60.2026s cumulative Host-observed preparation, while PLE gather/H2D dependency wait was0.000366s. This is a real synchronous provider boundary, but it combines canonical BF16 read, tensor staging/format work and device copy; it does not identify a raw DMA bottleneck.

Conditional next-round proposal: take one frozen layer0 routing group from the real replay/trace and separately measure canonical BF16 materialization, ND/NZ parent preparation and copy/retire using the same32-slot reservation. Preserve arithmetic, source identity and memory ceilings. This proposal is an attribution/decomposition experiment; it is not authorization to implement cache reuse or a new kernel now.

Rejected for this round:

- PLE launch/wait change: no evidence yet that exposed PLE wait is a material cost; no calibrated cross-stream overlap proof.
- CPU kernel rewrite: replay established one real-input numerical fixture, but native worker packing/GEMM/queue split remains unknown.
- GDN/QSA rewrite: actual eager providers identified, but sampled narrow-layer spans do not establish a full-model mutually exclusive bottleneck ranking.
- Shadow reuse/changed-slot/next-layer staging: total accounting bytes do not prove a safe reuse relationship or an admissible extra stage.

Final decision: NO_BEHAVIOR_CHANGE_SELECTED. The selected next-round proposal is EXPERT_SHADOW_PREPARATION_COST_DECOMPOSITION. No T5 PLE experiment was run. Product quality, formal performance, long context and deployment remain PENDING.


## Executable next-round experiment proposal (not started)

1. Freeze the same revision, MXFP4 bank, NZ29 layout, actual32-resident map and the captured layer0 top10 route fixture. Re-admit current shared physical15 resources; reuse is not permission to skip fresh admission.
2. At the existing prepare provider, separately observe canonical BF16 materialization, ND/NZ parent construction and H2D/copy completion. Record wall time, bytes, allocation lifetime and existing dependency events. Keep token scheduler chunk, expert microbatch and shadow lifetime as separate IDs.
3. Run one bounded component lifecycle with the actual expert set and one missing-expert transition; do not introduce another staging buffer or preserve a shadow across layers yet. Include flag-OFF reference and the real-input numeric oracle.
4. Accept evidence only when prepare subspans reconcile with the parent within recorded clock/observer limits, identities remain exact, output matches reference and resources return to post-construction ownership. If provider boundaries cannot separate costs, report UNKNOWN rather than label a memcpy bandwidth limit.
5. Choose a reuse/changed-slot or conversion improvement only after the measured cost and valid shadow lifetime support it. Separate approval/execution plan must bound any later full-model requests. This round's2model-start and3profile budgets are closed.
