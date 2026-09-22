# Synchronous PLE local gate — 2026-09-20

Status: PASS_PLE_LOCAL_ONLY. Not integrated into full model or SGLang request
pool. User now requires live execution through full-model testing; schedules
round9 and round7 are PAUSED. Continue without restarting them.

Implemented original-buffer N-gram hash with explicit immutable history,
CPU pread gathering from original128 BF16 shards, and eager per-branch
normalization/gating/dilated convolution with explicit caller-owned state.
No full95GiB PLE table copy or pinning, no async/prefetch/cache optimization.
Checkpoint layers.1 maps to configuration ple_layer_ids[2]. Original multiplier,
head vocabulary and offset tensors are used directly; no regenerated hash values.

Unit tests:4 pure hash cases,2 byte-reader cases; general suite48PASS.
Torch CPU:4 module tests PASS following a missing-module RED. Independent double
formula, empty input, invalid state, chunk/decode equivalence and immutable
input state checked. Stateless objects keep request histories separate.

Real source identity bound to frozen receipt,144live file sizes and config/index
hashes. Original128shard boundaries (256rows) exactly match safe_open slices.
Seeded token list including EOS, vocabulary endpoints and repetitions hashes
identically to the two unmodified upstream4ccff141 hash methods (AST-isolated).
Reference source SHA is recorded in device-v2.json and checked against local
frozen audit checkout; full upstream source remains in the remote evidence dir.

Physical15-only NPU probe:6configs (F32/BF16 ×1/3/17tokens),3repeat hashes agree,
plus tokenwise decode comparison. Synthetic hidden activations, real checkpoint
embedding rows and key/value/norm/conv weights. Frozen relative-L2 gates:
F32 0.0001, BF16 0.03. Maximum full-equation error F32 4.1055e-7,
BF16 0.00407788; max F32 decode difference6.4326e-8; BF16 decode/state identical.
Peak torch allocation145472000bytes; final allocated/reserved0.

First NPU attempt failed libgomp thread creation and aborted with secondary
pybind11/GIL errors (not a numerical failure). Container remained CPUquota2,
RAM8GiB, pids512. Setting only per-container OMP_NUM_THREADS/OPENBLAS_NUM_THREADS/
MKL_NUM_THREADS=2 resolved the failure; same math and thresholds passed. Preserve
failed device.log/container-failed.json and successful device-v2.* separately.
No host package/global environment/driver changes. Thread-limit explanation is
supported by the controlled rerun; no claim that every secondary error has an
independent root cause established.

Remaining: independently review row/state integration, request cancel/reuse
through actual scheduler, and connect synchronous PLE to full model; pure
caller-owned state tests do not qualify scheduler lifecycle. No model outputs,
CMMLU or full-model resource admission yet. Continue exact QSA and GDN gates.

## Review hardening

The probe now rejects reference-source bytes unless SHA256 matches frozen
4ccff141 qwen4_exp.py before AST parsing/execution. The constructor dilation
is explicitly bound to the validated hasher ngram_size. Rerun device-v3.json
passes all6 cases with unchanged tolerances and zero final allocations.
The prior v2 source hash independently matches the same frozen source.
Multi-request pool cancellation/reuse remains an integration gate.
