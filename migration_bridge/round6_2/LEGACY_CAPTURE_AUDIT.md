# Phase 1 — legacy capture association audit

Evidence: CODE_INSPECTED and read-only A3 raw-artifact verification. No model
inference was executed. New qualification output remains absent.

## Independent replay and runtime gate

Re-executed `tools/audit_legacy_replay.py` on retained B/C roots. Result:
436 common independent stage files, 436 byte-value exact comparisons, with
per-tensor SHA256 preserved. These files come from direct stage-local captures
in `experts_base.py` and `round6_1_deferred.py`, not the generic model hook.
Their association is SAFE for the compared stage/call sequence.

The final logits are a separate OFFSET_COMPENSATED artifact, not independent
outer-forward dumps. In both B and C, Pass00002 holds the 54-token request
input and positions 0..53. Pass00003 holds the response token at position54
and the preceding forward's logits. The reconstructed input IDs reproduce the
request_input_ids_sha256. API top-candidate relative logprobs independently
agree with captured logit margins within 2.04e-8. Both final logits hash to
98177a8cb38a920d0f70ead9ae60b947a9a56ed4342fbc16933fb32f89145c33.

The generic hook's ordering explains the compensation. The previous blanket
description "current pass input equals current logits" must not be used.
Evidence: `evidence/legacy_audit/replay-audit.json` (raw-file metadata and hashes).

Runtime buffer-immutability checks compare direct before/after-sync stage
files; pending ordering uses independently emitted JSONL sequence numbers.
They do not depend on final-logit dump association (NOT_APPLICABLE to this
specific defect). This retains their historical scope; it does not upgrade
them to new capture-OFF production requalification.

Phase 1 prerequisite is satisfied by the independent SAFE replay stage proof
and NOT_APPLICABLE runtime proof. The intended-approximation mechanism remains
supported; new capture and runtime gates must still be run.

## Artifact-by-artifact classification

| Artifact | Producer / association | Classification and scope |
| --- | --- | --- |
| Round6 QD full-logit matrices | round4a3 run_teacher_forced_sweep; selects positions[-1]==len(prefix), checks API selected token is a captured maximum; per-request new-file window | UNSAFE_FOR_NUMERICAL_REUSE until every original predecessor/input association is independently checked; not reused for V2 |
| Round6 HD matrices | same extraction | UNSAFE_FOR_NUMERICAL_REUSE under the same conservative policy |
| Round6 first_overflow | qualify_deferred_pairwise_contract, derived from HD matrices | UNSAFE_FOR_NUMERICAL_REUSE as independent numerical proof; historical gate record remains valid |
| Round6 matched_metrics | compare_deferred_matched_logits over saved matrices and manifests | UNSAFE_FOR_NUMERICAL_REUSE inherits unresolved source association |
| Round6.1 historical reproduction | same sweep versus old matrices | UNSAFE_FOR_NUMERICAL_REUSE for independent association; repeating the same extractor does not prove association |
| Round6.1 stagewise evidence | stage-local source/layer/stage/call files, direct tensor payloads | SAFE for historical aligned stage comparisons, not generic dump metadata |
| Round6.1 explicit replay stage proof | same direct stage files; independent explicit contribution storage | SAFE; 436/436 rechecked |
| Round6.1 explicit replay final logits | compare_stagewise_captures.find_logits uses request_input_length; predecessor and API proof added here | OFFSET_COMPENSATED; B and C verified |
| Round6.1 runtime/lifetime | analyze_runtime_controls; direct buffer captures and ordered trace | NOT_APPLICABLE to generic-logit association defect |
| Round6.1 numerical risk analysis | analyze_deferred_signed_risk plus summarize_deferred_risk, inherited QD/HD matrices | UNSAFE_FOR_NUMERICAL_REUSE for V2 calibration |
| Whole-request CPU-not-hit final-logit report | select_pass matches input_ids=[11], which is a response token rather than original input[1000] | UNSAFE_FOR_NUMERICAL_REUSE until explicit predecessor proof; do not label this as input[11] logits |
| Whole-request CPU-zero/merge boundary | independent stage tensors and route trace | NOT_APPLICABLE to generic-logit association defect; must be rerun for V2 |

UNSAFE_FOR_NUMERICAL_REUSE here means the required association proof is missing,
not that the reported numeric tensor values were proven wrong. No old QD/HD
tensor, statistic, failure or prompt is permitted to calibrate V2 or tune HD2.

## Plan reconciliation

The supplied execution plan names an earlier prequalification parent. Current
start is 47618943b9ce1e8ed100473059712854676c143f, a descendant containing
pre-output method/telemetry commits; child remains a903c004d8a78a93b5a57020f63f1e69525f0a67.
No QD2/HD2 output was produced during that difference. Existing preregistered
128/128, prompt-level risk, rank127 and alpha=0.05/3 agree with the new plan.
Shared NPU permission was explicitly granted separately and remains effective.
All old evidence files remain untouched; this audit supplements their meaning.
