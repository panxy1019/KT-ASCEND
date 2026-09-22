# Round6.6B final ADR — protocol change did not close the quality screen

Decision: **SCREEN_FAILED_NO_RESEED**. No production promotion or passing tag.
Both288-request collections completed normally and their owned processes were
reaped. The independent raw-response audit verifies all288 paired inputs,
decoding parameters,finite returned log-probabilities and reconstructed scores.
The collector's before/after91-file runtime and asset checks completed without
drift. This is not an infrastructure failure.

| Endpoint | NPU | Hybrid | Gains/losses | Invalid N/H | Truncated N/H |
|---|---:|---:|---:|---:|---:|
|GSM8K|14/32|13/32|2/3|1/2|2/1|
|CMMLU|174/256|173/256|1/2|0/0|0/0|

Frozen rule:zero observed net accuracy loss per endpoint,and no invalid increase.
Both accuracy endpoints fail;GSM8K also fails the invalid-count rule.
This is a conservative **engineering screen**,not a statistical noninferiority
test or proof that every floating-point-induced answer change is a runtime bug.
The zero-observed-loss rule was an explicit pre-execution study design choice;
it must not be misrepresented as a user-approved static2pp population margin.

## Statistical limits

Empirical paired20,000-bootstrap95% intervals for Hybrid−NPU:

- GSM8K:[−15.625pp,+9.375pp].
- CMMLU:[−1.953125pp,+0.78125pp].

Conservative paired-discordance exact-marginal intervals:

- GSM8K:[−26.85253pp,+21.55246pp].
- CMMLU:[−3.13014pp,+2.40113pp].

These data do not establish statistically significant population degradation,
and also do not establish population noninferiority. Neither interpretation
overrides the separately frozen observational screen. No post-hoc2pp allowance
or new seed is applied.

## Failure review

All correctness discordances and invalid outputs were inspected. The3GSM losses
contain actual wrong final answers (5→7,4000→1000,123→37),not correct answers
hidden by the parser. The2 gains are also real task improvements. The invalid
fractional outputs are inconsistent with their gold answers;accepting fractional
syntax would not recover the lost correct answers. No scoring fix is justified
by this review. CMMLU scores were independently recomputed from the four frozen
candidate-token log-probabilities,not from an arbitrary generated explanation.

The improved reference scores relative to V1 are not a controlled estimate of
prompting benefit:both the independent sample and protocol changed. This study
is official-style inspiration,not an exact reproduction of the official paper.
CMMLU invalid=0 is partly structural to candidate likelihood scoring and is not
evidence that free-form formatting errors disappeared.

Round6.6A had already established exact mixed-partial replay for all8V1 quality
discordances. This study does not turn that finite evidence into proof that every
possible longer operand is defect-free. No new violated implementation invariant
has been established. In particular,we have not performed a fresh full layerwise
capture of every new B discordance and do not label an unmeasured first-router
boundary as proven.

## Operational decision

The single preregistered protocol hypothesis is insufficient to close quality.
Do not continue changing prompts,seeds,masks,casts or acceptance rules merely to
obtain a favorable screen. No LLAMAFILE replacement,NPU execution of CPU-owned
experts,worker-count change or production arithmetic patch was made.

Historical V1 and Round6.5 remain BLOCKED. Four-R32 quality qualification remains
BLOCKED. All-MoE heavy qualification,Round7 Dynamic and the heavy Deferred probe
remain NOT_STARTED/NOT_RUN. Default Deferred remainsOFF.

The completed deliverable is the reference-lane implementation,operator/system
evidence and two independently frozen quality outcomes—not a successful heavy
offload deployment claim. A next study needs an independently justified change
or an explicit quality-risk contract;repeatedly testing fresh random subsets is
not a repair strategy. Small32-item GSM8K samples cannot resolve a tight
population noninferiority margin without substantial statistical uncertainty.
