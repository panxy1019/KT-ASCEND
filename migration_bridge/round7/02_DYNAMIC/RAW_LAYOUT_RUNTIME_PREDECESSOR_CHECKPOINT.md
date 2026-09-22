# Raw-layout / pinned-buffer runtime predecessor checkpoint

2026-09-17 23:01UTC inspection. A3_VERIFIED, not full Dynamic qualification.

Runtime-v6/tools-v17 `backing-requalification-v5` has closed:

- Operators100/100PASS, maximumrelativeL2 0.0017029279396063187 <=0.01,
  all six early/middle/late C32/C64 runs. Operator `tp=2` denotes two CPU
  subpools, not distributed NPU TP2; the model remains NPU TP1.
- Uniform32compact plus4restart/observer-OFF requests PASS: repeats, prefixes,
  restart, observer parity, finite, masks/mappings/ownership/layout audit.
- Lossless offline comparison: all four Uniform reference trajectories are
  exactly equal between runtime-v5 and runtime-v6. This is a bounded regression,
  not a universal cross-backend trajectory claim.

The restart server returned all4request responses, then the session deliberately
sentSIGTERM and its owned guard completed process-group cleanup. Transient
defunct children during teardown were inspected; process.log confirms deliberate
shutdown, lifecycle/reaping complete, and the parent driver advanced. The guard's
CANCELLED/-9 teardown record is not a new request/model failure. No unrelated
task was stopped or restarted.

## Lossless checkpoint

`raw/runtime-v6-uniform-checkpoint-v1.tar.gz`, SHA256
`6b0cb8281f978a6925d323bdb605b0d77b29490e124e708a254e551439619fe4`.
Full compact/restart process logs, raw requests, structure, lifecycle/reaping,
closed summaries, actual private wrapper/helper and provenance. Complete
SCPexit0, A3/local hash match, full tar validationexit0, JSON trajectory comparison
PASS. Detailed new operator arrays remain A3, not in this bounded archive;
historical complete operator/reference archives remain frozen separately.

## Successor

Healthy driver now RUNNING Frequencycompact, then directR64 and originalDEV
trigger/generation probes256/512/1024. Keep healthy run intact. No selected
threshold, candidate freeze, DynamicHOLDOUT, product quality, formal performance
or Deferredre-entry at this checkpoint. Future formal windows use independently
deployed hostobserverV3; no old contaminated Static window is reclassified.
