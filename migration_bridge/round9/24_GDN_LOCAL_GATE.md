# GDN real-weight local gate — 2026-09-20

Status: PASS_GDN_COMPONENT_ONLY, evidence device-v4.json. The GDN reference is
not yet connected to the SGLang model, state pool or scheduler. Full-model
resource admission and inference remain incomplete.

CPU tests5 PASS after missing-module RED. Physical15-only NPU tests18 PASS:
layers0/24/46 × FP32/BF16 × tokens1/3/17, with3 byte-identical repetitions.
Original checkpoint projections, convolution, A_log, dt_bias, normalization
and output projection; synthetic hidden states. A_log and recurrent state
retain FP32. Independent CPU FP64 equations cover complete output and both
state tensors. Tokenwise continuation and explicit-zero reset match prefill.

Frozen rel-L2 gates FP32=1e-4, BF16=.03, unchanged across failed/successful runs.
Maximum error across output/state/continuation/reset is1.087496400804069e-6
for FP32 and0.005065960661894483 for BF16. All errors must be finite. NPU peak
allocated258418176bytes; final allocated/reserved both0. Device isolation is
verified by ACL logical0→physical15, one visible device and only davinci15.

## Preserved transfer hang and bounded repair

v1 did not finish: the process remained active after the420s TERM timeout and
was stopped using only its owned container. v2 introduced periodic Python
stack dumps; it reached all three layer0 FP32 cases and then hung in the
next dtype's CPU→NPU weight transfer. v3 logged each tensor, locating the
stall at layer0 BF16 in_proj_qkv.weight. A hard timeout with KILL escalation
bounded the diagnostic run. No numerical FAIL was relabeled as PASS.

v4 separates CPU conversion from device transfer and clones each tensor into
a private CPU staging buffer. Each staged tensor is checked exactly against
the source values in the requested dtype before framework-managed .to(NPU).
Only one tensor's temporary host copy is added at a time. All18 cases then
completed under the same8GiB host limit,2 CPU quota and2 OpenMP/BLAS threads.
This controlled outcome supports retaining bounded private staging. It does
not identify a unique underlying CANN/torch_npu mmap/transfer defect.

v4 uses a separate immutable run-source snapshot on .3 under
/mnt/data1/kt-round9/run-sources/gdn-v4; local tested-sources-v4.json records
the corresponding implementation hashes. Failed logs and container states
remain under evidence/physical15/20260920-gdn alongside the successful result.

Remaining gates: actual model activations, upstream Ascend backend differential
comparison or a separately integrated eager reference path, packed requests,
state slot cancellation/reuse, KTEP ownership and full-model loading/inference.
The known upstream NPU sigmoid-vs-swish dispatch incompatibility is not fixed
merely by passing this eager module probe.
