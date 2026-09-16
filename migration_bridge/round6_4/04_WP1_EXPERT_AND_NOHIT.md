# Round6.4 WP1 missing controls — A3 verified

No production code changed. Evidence uses the actual checkout+overlay recorded
in each provenance JSON, not an assertion that the local documentation HEAD
was deployed unchanged. Other workloads and host libraries were untouched.

## Entire-request CPU-not-hit control

Frozen Round6.2 control SHA `b6911ac6122388529a81724ecde68bb4effe45bf595fba3b943e226ac1a904b1`;
input `[1000]`, output budget1. OFF and ON use separate fresh processes.
The retained Round6.3 capture tooling is reused only for transaction integrity,
not its rejected numerical contract. Required `r63-` server directory prefixes
are a launcher constraint and do not indicate old evidence relabelling.

Each process produced two physical forwards: one target and one lookahead.
Both final index audits report zero unknown/incomplete records. Every L7/L8
route in both forwards is NPU-owned, all CPU partials zero and all outputs finite.
Merge bytes equal NPU partial after the existing final visible cast. OFF/ON
full logits are byte-identical at both forwards. This proves a whole request
with a clean causal history, not merely a zero-hit row following previous defer.

Result: **WHOLE_REQUEST_NOHIT_PASS / A3_VERIFIED**.
Report: `evidence/r64-nohit-v1/whole-request-audit.json`.
SHA256: `f9aa82005897c30dac4bc85fb11c3176bb10f348707291b7839afb0c0af60c01`.
Raw tensors remain in the local mirror and A3 original paths; index and metadata
are versioned, binary tensors are excluded from Git.

## Real expert safety

All frozen D1 experts: L7{16,38,41,50}, L8{8,25,33,55}. For each, one decode
token and four prefill tokens; same deterministic BF16 hidden values and
checkpoint weights feed CPU and NPU reference. Gate/up shape1408×2048,
down2048×1408. F32 GGUF retains the existing BF16-valued source contract.

Reuse unchanged Round3 probe SHA
`4be8d94090a7cc5e16e9597ed2759494795b003472e9b6f27e9f80004bf4f3e4`.
Each case runs in an independent process. The retained probe assigns two old
loader class attributes; these do not alter production and process isolation
prevents reliance on a persistent loader reset. No old private lifecycle API
is used to score this test.

- 16/16 cases PASS; ten repeated CPU outputs per case have one hash.
- All CPU/NPU outputs finite.
- Maximum relative-L2: **0.003913061993474726**, safety threshold **0.01**.
- Guard reports EXIT_ZERO. Reports and source/library provenance archived under
  `evidence/r64-expert-safety-v1/`.

This is same-operands expert safety only. It does not demand equality between
Deferred and non-Deferred full-model graphs, qualify downstream quality, or
substitute for 1000 real deferred forward/lifetime validation.

## Promotion

These missing WP1 controls now permit full D1 runtime requalification.
They do not promote the complete Deferred feature to READY.
