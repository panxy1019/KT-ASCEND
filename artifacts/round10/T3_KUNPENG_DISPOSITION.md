# T3 Kunpeng Expert disposition

Status: `CURRENT_NATIVE_SAME_PRECISION_VERIFIED_CROSS_PRECISION_BOUNDARY_RECORDED_NO_CANDIDATE`.

The current provider path was reached below the Python submit/sync seam:
`KTMoEWrapper(LLAMAFILE) -> BaseMoEWrapper -> CPUInfer task queue -> native
llamafile MOE -> four 160-wide intermediate partitions on NUMA 2/3/4/7`.
The loaded extension SHA-256 was
`614ae60f70ff70d1e47d7fd32e857f887f874b6248eb332172c8a637cb471a6d`
and the layer0 MXFP4 bank SHA-256 was
`e9d6b27f352d501a9ed3737ec5d3ab140b9d1f5bda22d2e36ef9ef0bcc45b08b`.

On two real three-token/TopK10 layer0 groups, native versus an independent
software MXFP4 decode reference had relative L2 `3.27e-7` and `2.00e-7`.
Five repeats per group were hash-exact. Steady medians were 1.468 and 1.465 ms.
The same independent MXFP4 reference differed from the captured BF16 provider
contribution by 0.1290 and 0.1280. This exceeded the pre-run exploratory 0.05
cross-precision gate, so v100/v101 remain original FAIL evidence and no
BF16-equivalence claim is made. This does not invalidate the same-precision
MXFP4 implementation result or the T1 model-level Hybrid quality checks.

All six component process identities were consumed. v96-v99 exposed bounded
probe-launch contract defects before numerical evaluation; v100/v101 completed
native computation. No performance candidate was promoted. T2 showed a small
steady exposed CPU-result wait and the already-active C128 forward path is
numerically faithful to its MXFP4 oracle; no single-variable kernel change was
justified within the remaining budget. The fixed H0 CPU backend stays enabled.
Physical12 was empty after the final process.
