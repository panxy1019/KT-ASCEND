# Ascend BF16 resident-to-raw context repair

2026-09-17. Qualification in progress; no Dynamic promotion.

## Symptom and actual boundary attribution

The preserved v18 actual-model replay established that checkpoint-exact
resident inference matrices become incorrect during raw full-context Phase1.
Both Layer1 and Layer2 resident E0/E8/E56 failed after prepare, whereas CPU
staging controls E1/E63 stayed exact. Layer2/E0/w13 maxabs was 0.5719451904296875.
The full failed replay is preserved on A3 and has a verified local tmpfs copy;
see `WEIGHT_BOUNDARY_DIAGNOSTIC_ARCHIVE.json`. No failure is rewritten as PASS.

The deployed unquantized Ascend loader transposes w13/w2 after loading, whereas
the temporary full-context is allocated in raw orientation. The original
Phase1 assumed identical layouts. A separate actual-method negative control
already reproduced an inference-to-raw copy shape exception. These are weight
transport defects, not acceptable heterogeneous arithmetic perturbations.

## Minimal patch

Production candidate commit: `e7d51bf5309416ee5fcdc6c05f4d2e6750c60c1f`.
On NPU BF16 Phase1 only, convert the **full resident tensor** to ND before
slicing experts; transpose dimensions 1/2 when its audited shape is the reverse
of the raw destination. Make the normalized tensor contiguous, then retain
framework `copy_`, stream ordering and ownership. Reject unexpected shapes,
dtypes/ranks, or an internal-format full-context destination.

No checkpoint weights are injected; no GGUF values, expert selection, routing,
merge arithmetic, CPU staging, quantized or CUDA path is altered. The installed
torch_npu wheel and original deployed source remain untouched.

Private runtime-v7 wrapper SHA:
`86f0f1683f8ad00c96c5fd0c67fdc545631a292b5cc1c5e9a51bcedd91ca6dd7`.
Resident helper SHA:
`ac2ac86a25dfeef3548bea38e60305ad091eb9d83ae2d893b37250a65ffa106e`.
Repaired CPU extension remains:
`46182176279b072b4ab52d21bb8cc0ecf4d217399d486155c7fcd4f0690e21e2`.

## Local and A3 test-tool verification

104 pure-Python tests pass locally and under A3 Python3.11. Actual original
source guards pass. These checks are LOCAL_VERIFIED / A3 tool verification,
not A3_VERIFIED_READY for Dynamic.

tools-v21 source commit: `72de5bcc9d5508ddce497d7057e444ede4286540`.
Release archive SHA:
`1cbf2d133903af90c1b5fcef4b9065cdf35ab27251ed1453c36e77e9f26fd734`.
SCP completed with exit0, and host/container hashes matched before extraction.

## A3 regression and ordered recovery

Real-size Phase1 probes use hidden2048/intermediate1408 and nonidentity slot
mapping. Initial v19 requested NZ but torch_npu downgraded it to ND because
internal format was disabled; source format2 is reported explicitly. Those
results can establish ND correctness, **not NZ correctness**.

The v21 follow-up explicitly enables internal format only in the standalone
test and requires actual source format29. Raw and inference NZ cases must both
have exact raw output before proceeding. No test setting is inherited by serving.

Owned background chain:
`/workspace/c64-dev/round7/resident-layout-recovery-chain-v1.log`.
It waits for the existing owned primitive matrix to exit (never interrupts it),
checks the three fixed ND results, executes the two actual-NZ tests serially,
checks their exactness, and only then starts the fresh predecessor driver:
`/workspace/c64-dev/round7/backing-requalification-v6/progress.json`.

Fresh ordered driver: operators → Uniform compact/restart → Frequency
compact/restart → direct R64 required174/256 and exact historical CLI → original
DEV trigger probes256/512/1024. No old PASS summaries copied to runtime-v7.

The diagnostic auditor now fails immediately on captured prepare-entry/output
checkpoint mismatch; it stays OFF in formal timing. After all three probes close,
continue DEV threshold screen and commit the candidate before held-out use.

At this document's checkpoint, actual-NZ and full-model recovery are **PENDING**.
No threshold selected, no Dynamic HD/quality/performance, no Deferred evaluation.

## 2026-09-18 00:06–00:08 A3 checkpoint

Actual format29 raw and inference sources both PASS, w13/w2 raw identity exact,
maxabs0. The fixed ND jobs also PASS; the old requested-NZ/actual-ND distinction
remains visible in the raw reports rather than being rewritten.

Lossless primitive matrix, private source/provenance and release archive:
`raw/resident-layout-nd-nz-regression-v1.tar.gz`, SHA256
`c9685a6b740b9e0e1eba29726c03841407810733561fae7905e6ad08645c0285`.
Complete SCP exit0, matching A3/local SHA and full tar validation exit0. Size
about2.3MiB. This is A3_VERIFIED for the isolated transport regression, not
A3_VERIFIED_READY for Dynamic full-model migration.

The background chain automatically entered fresh `backing-requalification-v6`.
Operators100/100 PASS, maxrelativeL2 0.0017029279396063187. Uniform first-process
32/32 compact requests finished and were gracefully drained; its separate clean
restart is loading/running. Uniform gate is not yet promoted without that result.
Healthy run retained. Frequency, directR64 and DEV trigger gates remain pending.

A separate owned, flock-protected watcher
`/workspace/c64-dev/round7/dev-threshold-screen-auto-v1.log` will run DEV-only
threshold_sweep only if the driver's exact complete-predecessors status appears.
Failure/timeout or any missing trigger PASS prevents fitting. It cannot create
a commit acknowledgment or launch HD/quality/performance. The agent must audit,
commit the eventual candidate and verify commit contents before those consumers.
