# Round9 full-model resource and fresh reference audit

## Execution scope

Host 110.123.0.3, physical15, shared training PID4069720. User attested a15GiB
training workload envelope including validation/save; this is not an enforced
reservation or an independently measured training peak. Frozen original revision
`de4b8e4d43b917e7706784d8bb445c9af86a3540`, Q64 BF16 routed storage with F32 CPU
compute/partials, C64, text TP1/batch1/inflight1. Internal context4098 and pool4100
provide an actual4096-token workload after f02 scheduler guard slots.

## Completed v5 resource/system gates

- Actual3840 input +256 generated tokens PASS, finite output logprobs.
- Clean restart4 and four short requests after the long request match v3 compact32.
- Owned-process HBM peak39850MiB; board peak48708MiB.
- Torch allocated peak40705233920bytes; reserved peak41502638080bytes.
- Container memory peak303213101056bytes; memory failcnt0.
- Training process6124MiB throughout sampled execution.
- v5 stopped intentionally after these tests; training untouched.

Source receipts are under `evidence/physical15/20260920-full-model-v5/`:
`context4096-v1.json`, `restart-v1/summary.json`, `post-context-v1/summary.json`,
`resource-measured.json`, `monitor.jsonl`, `container-final.json`.

Qualification is the observed loading and synthetic4096 resource workload for
this exact profile. It does not establish long-context semantic quality, a hard
shared-device reservation, or performance qualification. The old static preflight
receipt remains unchanged; no retrospective conversion of that receipt to PASS.

## Fresh quality reference in progress

v6 observerOFF uses the hardened monitor (stable3-sample prestart baseline,
fail-safe own-container stop even if log output fails). FirstHTTP and restart4
passed. Frozen CMMLU256 runs as two contiguous128-item slices with identical
model/runtime/profile/driver. ManifestSHA
`88efae7d68e7edf644add12909db89712dc001046b8624801dbb8ce7a8804858`
was committed before scoring in `e66726f2f6a07aad85a974a15e0aba7604a20852`.
No previous-model scores are inherited. Final raw-response audit remains pending.

Before v6 scoring, commit34efa6551 bound the exact freeze digest/commit,
image/profile/implementation and current-container restart receipt.
An initial client launch used a nonexistent /workspace path and failed before
any request; corrected to /round9/src. No model/runtime change was required.

Local validation:23 admission/monitor regression tests PASS. Timers remain PAUSED;
no push, merge, tag, or canceled Round8 performance work.
