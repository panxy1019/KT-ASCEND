# Round6.4 fast quality results

Evidence: A3_VERIFIED, frozen-subset engineering screen only. Both independent
OFF/ON processes completed288/288 requests, normal planned stop. No production change.

| Benchmark | N | OFF correct | ON correct | Delta | Gains/losses | Screen |
|---|---:|---:|---:|---:|---:|---|
| GSM8K native |32|5 (15.625%)|5 (15.625%)|0pp|1/1|PASS|
| CMMLU |256|139 (54.296875%)|139 (54.296875%)|0pp|0/0|PASS|

Per-benchmark observed delta >= -2pp under the user-approved minimum-cost protocol.
Identical counts do not imply identical answers: GSM has one correctness gain and one loss.

## Limitations retained

- GSM length-cap truncations OFF21/ON18; invalid0/0. All truncations count wrong.
- CMMLU invalid7/7, truncations0/0; all invalid outputs count wrong.
- GSM paired marginal interval diagnostic [-0.18304209,0.18304209]; inconclusive
  for population noninferiority. Its bootstrap95% [-0.09375,0.09375] is supplemental.
- CMMLU corresponding interval [-0.01697162,0.01697162] under the working paired
  model; zero-discordance bootstrap[0,0] does not establish zero population risk.
- No interval gates or sample expansion after observing these results.
- Low GSM baseline/truncation limits sensitivity; no broad mathematical quality claim.
- Literal overlap:1/32 GSM with historical material,0/256 CMMLU; no dev literal overlap.
  This is not a model-pretraining contamination audit.

## Integrity

Offline re-audit checks288 unique IDs, exact frozen input IDs/order/stop/budget,
prompt lengths, finite validated outputs, re-scoring, actual source hashes,
binding to protocol, and normal lifecycle termination. Both PASS.

- Input SHA: `de64f796d530b649b7e1dc7854aebef441be65cdf60762a571305450ad7fbc7c`
- Protocol SHA: `19df819323339c50ae495a5658e89f8cbef2dcb5d8881a00e9de9973ea148184`
- OFF raw SHA: `f64efd17427f1dfebc4f18d58179195327f67619deaf9cf0f37381ecdc12ad19`
- ON raw SHA: `05d59eae18484ee9435ea7e41ac7c0f5246a37e0aac3bbcb530aa614243c9346`

Evidence: evidence/r64-fast-quality-{off,on}-v1/, fast-quality-{off,on}-audit.json,
and r64-fast-quality-summary-v1.json. Runner deployment /workspace/results/r64-fast-v1.
Generation seconds GSM1590.75/1593.04, CMMLU180.63/171.21; these unequal-output
quality timings are NOT the formal paired fixed-length performance result.

## Decision

QUALITY_SCREEN_PASS. Continue the already frozen minimum performance and remaining
runtime/held-out audit. Neither SPEEDUP_VERIFIED nor full Round6.4 READY is established here.
