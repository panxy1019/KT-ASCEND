# Repaired-runtime predecessor checkpoint

2026-09-17 19:45–19:50 UTC. Evidence: A3_VERIFIED.
This checkpoint is not full-model Dynamic qualification or Round7 completion.

## Representative local operators: PASS

Same frozen real operands, weight source, NPU/canonical references, numerical
boundaries and0.01 safety gate; repaired CPU native SHA
`46182176279b072b4ab52d21bb8cc0ecf4d217399d486155c7fcd4f0690e21e2`.

| Layer | CPU workers / subpools | Cases | Maximum recorded relative-L2 |
|---|---|---:|---:|
|1|32 /1|22|0.0009405782341776842|
|1|64 /2|22|0.001673673779008822|
|13|32 /1|6|0.0012146409043853155|
|13|64 /2|6|0.0017029279396063187|
|26|32 /1|22|0.0011183248339594006|
|26|64 /2|22|0.0016933269530168054|

All100 cases PASS, finite, ten-repeat exact, CPU-not-hit exact zero. C64 cases
pass native FP32 partial merge exactness and sampled strided weight identity.
Cases cover experts1/63 with three historical decode operands. Layers1/26
also use stacked decode rows at qlen2/8/9/10/11/16/32/64: these are shape-boundary
diagnostics, not true contiguous captured prefill sequences or exhaustive experts.

Lossless archive `raw/post-fix-operator-evidence-v1.tar.gz` preserves **all**
operator files, including weight/partial/tensor dumps, six JSONs, logs, summary
and repaired imported-file provenance. SHA256:
`142e187455ef6723b031cab984c13610eba12d1321c7f593b76fe091dee6986a`.
Created only after operator subprocesses ended. SCPexit0; local/A3 hashes match;
full tar validationexit0. Original operator/static archives remain immutable.

## Uniform compact: PASS

Repaired runtime-v3 independently completed32 compact requests and four
clean-restart/observer-OFF requests. Same-path repetition, prefixes, restart and
observer parity exact; finite, all26-layer mapping/ownership/BF16 layout PASS.
The four reference trajectories also exactly match original runtime-v2 compact
trajectories. This supports no Static arithmetic change, not general model-level
bitwise equivalence.

A3 summary: `runtime-v3/uniform-system-summary.json`, SHA
`71af0dcb55d5c15e15cb91f7568646050fb7624f1532029aa03cb50f3a48f657`.
Structure SHA
`4751c642be1bb1477b9f2a9dfa476774eaf8ce576ce0653fcefbfff3087b6ee5`.
Full Uniform process/raw/preflight/lifecycle evidence remains at A3; local
lossless serving archive will be frozen after both compact lanes close.

## Healthy next stage

Driver `backing-requalification-v1` is RUNNING, phase frequency-compact,
completed stages operators/uniform-compact, no reported error. Preserve it.
Next Frequency compact→fresh R64 CMMLU256→three DEV migration probes.
No threshold candidate fitted/frozen, no Dynamic HOLDOUT or quality run yet.
