# LLAMAFILE staging runtime predecessor checkpoint

2026-09-17 22:02UTC. Evidence state A3_VERIFIED, not triggered migration
qualification or Round7 completion.

## Closed on runtime-v5

- Operators100/100 PASS: all six C32/C64 early/middle/late regressions,
  maximumrelativeL2 0.0017029279396063187 <=0.01. This protects CPU expert
  arithmetic, not full-context model integration.
- Uniform compact32 requests plus four independent restart/observer-OFF
  requests PASS: repeat, prefix, restart, observer parity, finite,
  mask/mapping/ownership and26-layer layout audit.
- Offline lossless-archive comparison: all four reference trajectories exact
  vsruntime-v4 Uniform. This is a bounded Static regression, not a universal
  arithmetic/trajectory equality assertion.

SummarySHA `9d95e3593e3dd907e940f3998932bfe8041c516e4f320cbbcdd7ee0b2be81107`.
Private wrapperSHA5762b338... and stagerSHA b623670d... are fully bound in the
separate staging compatibility document/provenance; CPU native unchanged461821... .

## Lossless checkpoint

`raw/runtime-v5-uniform-checkpoint-v1.tar.gz`, SHA256
`162d4a2b91835cd3b91f1dbf21cdd4edc025365ef07d828b13ca690c59310dce`.
Contains full Uniform compact/restart process records, requests, structure,
guard/reaping, provenance and closed operator summary plus driver progress.
Complete SCPexit0, local/A3 hashes equal, full tar validationexit0. Detailed new
operator arrays remain A3 and are not included in this bounded archive; earlier
complete operator/reference archives remain frozen separately.

## Healthy successor

`backing-requalification-v4` remains RUNNING Frequency compact on tools-v16/
runtime-v5, then directR64 and original DEV probes256/512/1024. No model failure
reported at inspection. Keep it intact; do not copy old PASS summaries, select
threshold on partial results, or launch HD/quality before candidate freeze.

Future host-observerV3 is only prepared locally (commit918b5256...,
sourceSHA9e92b9cd...). ActiveV2/current model tools are untouched. V3 needs
independent A3 deployment/smoke/fresh windows before formal use; old Static
campaign remains PENDING_HOST_ISOLATION. Neither the new tests nor V3 preparation
certifies Dynamic weight/mapping generation publication or default enablement.
