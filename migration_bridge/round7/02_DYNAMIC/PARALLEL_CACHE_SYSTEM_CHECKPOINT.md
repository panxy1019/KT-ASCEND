# Cache runtime-v8 parallel system checkpoint

## Completed join — 2026-09-18 04:26 UTC

All256/512/1024 full probes PASS. Both relocated worker chains exit0; each
has Process A5, Process B2, observer-OFF4 requests, COMPLETE lifecycle and
complete owned reaping. Host join checked imported artifacts, GGUF, frozenDEV,
profile identity and distinct threshold identities before accepting results.

Joined summary SHA256:

- 256: `1c1c5f3b0c240607982bd656965d2f74e528a29fa3f3542c81c4c37c08acdb1d`
- 512: `2ee2432c1ed72f6466ba74aa561d907063ee024c001930f5a8b6a343f111a48b`
- 1024: `de2d0c6eed67832c6fdb18a92f2200914981cd7a8719b0447356a518747bb85d`

Full512/1024 raw and joined manifest archive:
`raw/round7-parallel-probes-complete-v1.tar.gz`, SHA256
`58447f0122018efa17c052bbf21ca9139a6148e601e68fa65c213572b1b5a09b`.

Original worker now runs serial `dev-threshold-screen-v2`; new workers idle.
Concurrent probe durations are excluded from threshold selection/performance.
No candidate frozen and no held-out/quality launched at this checkpoint.

## Earlier checkpoint

2026-09-18 04:07 UTC, A3 execution evidence. Not threshold selection, quality,
or performance qualification.

## Completed

- Original Uniform compact32 + restart4: PASS.
- Worker A (NPU4, CPU0–63, NUMA0/1) compact32 + restart4: PASS.
- Worker B (NPU6, CPU64–127, NUMA2/3) compact32 + restart4: PASS.
- All three compact reports: same-path, prefix, restart, observer parity,
  finite and mask/mapping/ownership true.
- Dynamic256: PASS, Process A5 requests, B2, observer-OFF4; all three
  lifecycle reports COMPLETE, owned reaping complete.
- Dynamic256: exact migrated weight identity; generation consistency;
  deterministic selection; restart; no-trigger and trigger; observer parity.
  Workload shift changed resident sets in26/26 layers.78 diagnostic updates,
  10,796,138,496 diagnostic migration bytes. These synchronized audit durations
  are NOT production performance measurements.

512 and1024 were running Process A (two completed requests each) at checkpoint.
No held-out or quality launched; preserve healthy workers.

## Provenance nuance

Worker-specific `cpu_set`, `numa_nodes`, `container`, `physical_npu` and actual
resolved argv identify relocated execution. Copied provenance's legacy
`cpu_affinity` field still reflects original source128–191; it must NOT be used
to claim actual worker affinity. The inherited source provenance is explicitly
bound by SHA. Preserve active files; the preparation utility now corrects that
field for future preparations. Actual argv and topology checks showed A0–63/
nodes0,1 and B64–127/nodes2,3. This is metadata remediation, not a kernel change.

## Preserved archive

`raw/round7-parallel-cache-checkpoint-v1.tar.gz`

Host SHA256:
`181a760c2407daec6367b27349da82aa27458f174c8bbedd9b8611270a3e1876`.
Contains full256 probe, original compact summary, A/B compact/restart raw
results and worker provenance.512/1024 unfinished raw files are NOT included.

## Next

Join all three PASS with artifact/corpus/mask identity and completed reaping,
then isolated DEV selection, local candidate commit and ACK, then parallel
held-out/quality with ordered prerequisites. Formal performance remains isolated.
New local `operator_evidence.py` explicitly binds unchanged CPU arithmetic
evidence for later quality (`--operator-evidence-base runtime-v7`); it checks
native files, model/GGUF/software identity and safety gate. It does not create
a new operator PASS execution or replace cache/migration gates.118 local tests
PASS; this quality adapter has not yet been deployed or exercised on A3.
