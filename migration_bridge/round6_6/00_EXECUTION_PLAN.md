# Round6.6 — independent heavy-static numerical qualification

Status: IN_PROGRESS. B0 and QD6 audits passed; fresh52-request ten-repeat/prefix and4-request clean restart passed. Candidate frozen atSHA25643a0bd582a4b126feb7ed851b85e1e94c1de98131700547eead4615d3e21967a. HD6 one-shot capture RUNNING; quality NOT_RUN. No final promotion.

Base: `a0a233e3779089c1e767f689b981c2563ab10871`.
Branch: `codex/round6-6-three-way-numerical-qualification`.
The pre-existing dirty `third_party/llama.cpp` is excluded from this work.
Round6.5 remains BLOCKED; Round6.5A's bounded root-cause conclusion is retained.
No historical tag, fixture, mask, numerical bound, or result is overwritten.

## Execution checkpoints

1. Verify external primary sources and freeze Phase -1 operand selection before comparison.
2. Three-way expert comparison: actual Ascend GMM, installed LLAMAFILE, independent CPU FP32 reference. Add BF16-boundary and final-only-cast reference controls.
3. Cover four layers × 32 CPU-owned experts, three real decode inputs per expert; explicitly distinguish naturally routed operands from real-hidden unrouted controls. Real prefill qlen coverage is reported separately, never inferred from reshaped decode rows.
4. Resolve local numerical anomalies and true-production projection observability before B0 closure. A reference decomposition is not an LLAMAFILE intermediate capture.
5. Freeze independent Heavy Static Contract V1 and QD6/HD6 together. No observed root-cause prompt may become final heldout.
6. Expanded local qualification → QD6 → candidate freeze → one-shot HD6 → paired quality → ADR.
7. A failed hypothesis gets a separate version and new heldout if needed; autonomy is not permission to keep tuning the same heldout until it passes.

## Phase -1 protocol v1 (pre-result)

Frozen historical source: `/workspace/results/r65-four-R32-localization-v1/hybrid/stagewise`.
Layers: 1,9,17,26. CPU logical IDs: every odd ID in [0,63].
Decode calls: 2,33,55, individually executed with qlen=1 (not a qlen=3 batch).
These are real captured layer operands. Every CPU expert receives these same three inputs for complete coverage, whether or not naturally routed. Natural route membership and weight are separately recorded; this is not 128-expert production-hit evidence.
Prefill initial control: actual call1, all 11 original rows retained. Additional real prefill buckets remain required and cannot be claimed from this one bucket.
Additional prefill protocol (before additional prefill outputs): new operator-only prompts with 16/64/128 input tokens, one generation token each, calls1/3/5 in capture. Sample eight fixed CPU experts per layer: {1,9,17,25,33,41,49,57}; four layers give32 instances ×3 buckets =96 cases. This uniform sample is chosen without observing these operands' numerical outputs. Keep all original rows per prefill, never assemble synthetic prefill from decode rows.
Weight source: frozen model BF16 weights exact-expanded to FP32; installed production loads frozen F32 GGUF. Preserve prior full weight-identity proof and record fresh per-projection hashes.
Repetitions: 10 per production expert/input; nonfinite or nondeterminism fails immediately.
B0: any production-vs-pure-canonical CPU relative L2 >0.01, or unexplained projection anomaly, blocks model qualification pending investigation. Also retain production-vs-GMM 0.01 safety. Ratios are descriptive, not fitted thresholds.
All reference stages and final output are saved with operand hashes. Pure FP32, BF16-visible boundaries, and final-only BF16 cast are separate named semantics.

## Resource discipline

Target 110.129.0.22, NPU0 only, dedicated `kt-r65-static-offload` container, CPU0–31 / one CPUInfer pool. No host libraries or other container changes. Fresh preflight found NPU0 without a process, 986GiB host available, swap use zero; dedicated container stopped. This is an observation, not a persistent resource reservation.
Runtime source/extension hashes must match the Round6.5A actual deployment; local branch HEAD is not the installed runtime SHA.

## Exit evidence

B0 was NOT_RUN at preregistration and is now target-verified; see04_B0_FINAL_AND_STUDY_FREEZE.md and evidence/b0_decision_v1.json. Missing production stage capture or missing real prefill buckets must remain explicit, not silently waived. Production changes require reproducible semantic defects and dedicated regression. Static quality criteria must be independently preregistered; Deferred's historical2pp concession is not silently inherited.
