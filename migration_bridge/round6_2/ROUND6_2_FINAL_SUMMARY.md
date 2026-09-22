# Round6.2 gate-stop summary

**BLOCKED_INSUFFICIENT_EXPOSURE — not a completed qualification.**
Decision: `ADR_QD2_INSUFFICIENT_EXPOSURE.md`.

## Repository and deployment

Local branch:`codex/round6-2-deferred-contract-v2`.
Capture integrity freeze:`589c7defa2a7b88773d735fbb91de1e8e126898e`.
Local SGLang:`0a667fee2afd4b2d65fa6e4bff13a714f1407f37`.
Actual A3 checkout parent:`e3084d309d6f39b8a9f7185d36ac27e4b77526d7`;
child:`eaa58ccea4ee372b8b207dd16f398eafe3da37ff`, plus explicitly hashed
source overlays. Local HEAD is not the unmodified A3 checkout.
Complete executed source/library hashes are in retry1-provenance.json.
Inherited dirty llama.cpp was preserved untouched.

Container:`kt-r6-deferred-ascend`; NPU0 only;CPU affinity0–15; CANN9.0.0,
Python3.11.15,torch2.9.0+cpu,torch_npu2.9.0.post2. No host-library or business
container mutation. Model revision:`604d5664dddd88a0433dbae533b7fe9472482de0`.
D1 layers7/8; frozen model/tokenizer/placement/GGUF/corpus/history unchanged.

## Gate matrix

| Gate | State | Evidence |
| --- | --- | --- |
| Legacy capture audit |PASS in documented scope|LEGACY_CAPTURE_AUDIT.md|
| Complete-forward telemetry |A3 smoke verified|05_CAPTURE_INTEGRITY.md|
| OFF/reference vs ON smoke |20/20 byte exact per mode|evidence/smoke/|
| Method and input freeze |Before model outputs|CONTRACT_V2_SPEC.md; corpora/|
| QD2 defer0 |8192/8192; audits PASS|evidence/qd2-defer0/|
| Initial defer1 |INVALID startup contamination|QD2_CAPTURE_INCIDENT.md|
| Boundary correction |20 requests/40 forwards PASS|evidence/qd2-capture-incident/|
| Fresh defer1 |8192 positions/16384 forwards PASS|evidence/qd2-defer1-retry1/|
| Deferred transitions |8192/8192 PASS|transition_audit.json|
| Actual Deferred exposure |**343<500 FAIL**|evidence/qd2-exposure-gate/|
| Numerical reserve derivation |NOT_RUN; metrics unread|exposure_gate.json|
| Candidate freeze |NOT_RUN|no candidate produced|
| D1 runtime / HD2 |NOT_RUN|predecessor failed|
| Quality / free generation / performance |NOT_RUN|predecessor failed|
| ADR |STOP_INSUFFICIENT_EXPOSURE|ADR_QD2_INSUFFICIENT_EXPOSURE.md|

QD2 capture succeeded after infrastructure correction, but did not exercise
enough Deferred positions to qualify the frozen experiment. This is neither
a measured numerical failure nor evidence of numerical compatibility.
No threshold fitted, held-out output observed or favorable prompts added.
No READY tag or GitHub push is claimed.

Fresh sweep took about103.5minutes including final audit/client cleanup.
This is instrumented capture duration, not a performance A/B or speedup claim.
Only client boundary/readiness test tools changed during the documented retry;
model production code and numerical rules did not. Old failures/new successful
audits remain separately preserved.
