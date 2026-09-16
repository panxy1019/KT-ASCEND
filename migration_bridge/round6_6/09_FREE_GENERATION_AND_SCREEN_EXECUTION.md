# Observer-OFF free generation and frozen screen execution

Target:110.129.0.22,910B3,NPU0,32workers,one CPUInfer pool; production overlap.

## Free generation (target verified)

12/12 HD6 prompts ×64 tokens completed,finite,clean owned-process teardown.
Every path agrees with its observed teacher-forced natural sampler decisions
through the complete common history including the first cross-backend divergence.

10/12 trajectories exactly match All-NPU. Differences:

| Prompt | First token index | NPU / Hybrid top1 | Pairwise distortion at first difference | Historical diagnostic |
|---|---:|---|---:|---|
|r66-hd6-02|24|20177 /14859|0.125|Ambiguous,member,not stable|
|r66-hd6-06|47|207 /185|0.125|Ambiguous,member,not stable|

Generation artifact SHA:
`a6154705d63216161abc42779694da765d79c9b91ef0e746eba5bc9a90c9a029`.
This is not a quality score or population-equivalence proof.

## Screen execution

Owned serial sequence `/workspace/results/r65-r66-final-screen-v1`:
freeHybrid → common-history audit → All-NPU quality → four-R32 quality → paired
summary → human-readable engineering ADR review. Execution started after the
hashed HD/tail review; exact corpus and evaluator remain frozen.

Dependency supplement, inspected before quality answers were available:
`paired_quality_interval.py` target SHA matches the retained repository file:
`74eed4fc7ddf2f7831bc76a7c739ec22e6c849b4f54c52f71fd46b18c025db64`.
The new comparator imports **bounds() only**, not the old Deferred evaluate()
with2pp margin. Exact marginal intervals use union-bound coverage; a separate
20,000-sample paired bootstrap is descriptive. No2pp acceptance is inherited.

`summarize_final_screen.py` is postprocessing only. It reports subject counts,
invalid/truncation counts,Wilson accuracy intervals and request throughput while
keeping final_qualification=false until ADR. Small subject cells cannot support
subject-wise noninferiority. Throughput includes prefill and depends on generation
length; it is not a controlled decode speedup comparison.

Local regression:16/16 stdlib tool tests PASS. These are LOCAL_VERIFIED only,
not a substitute for target execution.

At this checkpoint quality results were pending. The completed screen subsequently failed; see12_QUALITY_ADR.md and evidence/quality_v1. No PASS is inferred from successful collection.
