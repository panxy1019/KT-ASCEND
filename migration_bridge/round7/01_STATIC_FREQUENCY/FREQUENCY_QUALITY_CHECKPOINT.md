# Static Frequency CMMLU product checkpoint

Evidence state: **A3_VERIFIED**, complete frozen256-item product screen.
This closes Frequency quality, not performance or final scheduler selection.

| Profile | Correct | Accuracy | Delta vs frozen R64 | Invalid | Product gate |
|---|---:|---:|---:|---:|---|
| Frozen R64 | 174/256 | 67.96875% | reference | 0 | reference |
| Frequency R8-equivalent C64 | 173/256 | 67.578125% | -0.390625pp | 0 | PASS |

The unchanged observed-5pp rule requires >=162/256 and invalid0. Same immutable
CMMLU manifest, input IDs, letter-candidate-logprob evaluator, tokenizer and
generation protocol as the reference. No model/arithmetic/merge/router/weight
loader/shared production path was changed in this static phase, so the frozen
R64 reference can be inherited. This is not a fresh R64 reproduction; a future
common-loader fix must reproduce R64 before using that repaired runtime.

Paired correctness diagnostics after validating identical256 unique IDs:
both correct170, both wrong79, R64-correct/Frequency-wrong4,
R64-wrong/Frequency-correct3. These do not replace the frozen product criterion,
establish statistical equivalence, or serve as an expert-selection objective.
Do not choose or refit masks according to these correctness outcomes.

## Evidence provenance

CMMLU manifest SHA256:
`bca1e1ec41da188e51f176813e6d916120eaab26b17a3dc63a99ed712cf9bf04`.
Original R64 quality artifact SHA256:
`d8bb0db8c2450b9e5f29608a930a638399ea90218a939dc05088055aa50d882a`.
Lossless archive: `../raw/frequency-quality-evidence-v1.tar.gz`.
SHA256: `22793c6a95c543a10664fc265769d77aaad4539f069512ee2c968ce0e3590038`.

Archive retains all raw Frequency requests/responses and scores, original frozen
R64 quality JSON, structure, resolved command, preflight, logs and lifecycle.
Closed Frequency quality lifecycle=COMPLETE and reaping.complete=true. Remote
archive creation completed before transfer; SCP exited0 before local checksum,
all-member archive read and paired-ID checks. A3/local archive SHA match.

## Ordered continuation

Healthy detached release-v6 driver automatically entered
`uniform-vs-frequency-performance`. Observer OFF, three ABBA blocks per bucket:
HOLDOUT_000 (71-input/256decode), HOLDOUT_032 (503-input/512decode),
HOLDOUT_080 (3511-input/1024decode);36 measured requests total.
No Static baseline choice before timing/window audit. Independent host observer
V2 continues; any incomplete/contended actual measurement window must not be
claimed formal isolated performance. Single EXCLUSIVE samples are insufficient.

Only after static serving fully finishes/reaps: original-native CPU backing
eviction MRE, minimal repair if reproduced, required regressions/reference,
then Dynamic DEV-only threshold selection/freeze and independent evaluation.
Deferred remains OFF. Round7 is not complete.
