# QD3 v1 rejection — preserved, investigation in progress

## A3 evidence

All three capture/integrity audits passed; self-history byte parity passed.
8192 target positions,7390 exposed. Frozen derivation ended REJECTED at
Unix1789493384.4708831. No candidate.json was emitted. Runtime,HD3,FD3,
quality and batch2/4 remain NOT_RUN.

Raw report remains on A3:
`/workspace/results/r63-qd3-v1-derivation/report.json` (33019269 bytes).
Raw SHA256: `7297cbbbac5ccd20901190e92e98724d20da2d07e4c73bbd0fc76dc5e17a23f9`.
Canonical SHA: `fea1bfb82b7da91e2c5f5834173eab9ac74f400eb656b4234eccce8dba9eea6e`.
Local working copy: `/tmp/r63-qd3-v1-rejected-report.json`; durable compact
extract: `evidence/qd3-v1-rejected-pairs.json`.

The registered rank127/128 reserves were2.125 for anchor contraction,11.25
for full-logit max absolute error,0.3464996752100513 for relative L2.
Each tail had1/128 exceedance and upper confidence bound0.046320923824670394
under the registered0.05 budget. These are REJECTED derivation outputs, not
an approved contract. There were4647 stable positions and0 stable top1 flips.

| Position | Candidate pair | Baseline margin | Deferred margin | Top1 |
|---|---|---:|---:|---|
| qd3-math-11:43 | 11,825 | 2.25 | -0.75 | 380 unchanged |
| qd3-math-19:56 | 64,28 | 2.875 | -0.125 | 87 unchanged |
| qd3-math-19:56 | 64,1366 | 3.0 | 0 | 87 unchanged |

Both rows are PREFIX_ONLY_EXPOSURE. Every listed pair excludes the anchor.
The frozen rule counts a positive-to-zero margin as a failure as well.

## Analytic issue to investigate, not a rule change

Let delta_i = (z_a^D-z_i^D)-(z_a^N-z_i^N).
The registered contraction statistic is max_i max(0,-delta_i).
For arbitrary candidates i,j, the pair-margin distortion is delta_j-delta_i;
its contraction is delta_i-delta_j. A bound on negative delta alone does not
bound this difference. Even bounded positive and negative components combine.
The evaluator uses the anchor-contraction reserve for all candidate pairs.
This mismatch in what is measured versus what is guaranteed is analytically
demonstrable, but the explicit frozen all-pair gate still correctly rejects
these observations. Do not silently change it to anchor-only or double it.

## Official-source comparison and limits

Official source was searched again after the failure. Pinned source inspected:
[KT experts_base at0dce4c9](https://github.com/kvcache-ai/ktransformers/blob/0dce4c9b097710951d1b3e232f50b973ff0a5505/kt-kernel/python/experts_base.py).
Its runtime contains callback/subscriber and device-wait handling; those are
runtime mechanisms, not evidence that this project's all-candidate ordering
gate can be relaxed. Existing route-only audit established selector agreement.
No upstream fix has yet been shown to repair these two prefix-history cases.
No production root cause is established from final logits alone.

## Next minimal experiments

1. Independently recompute the failed margins from the four original capture
   tensors and compare to report hashes; preserve exact input/history joins.
2. Inspect producer/consumer transaction traces for these complete prefixes,
   including pending drain, exactly-once and buffer identity. Current-step
   no-defer is insufficient because previous deferred steps affected KV history.
3. Compare local merge/consumer semantics to pinned official code and the
   existing Round6.1 explicit-replay controls; reproduce only these fixed
   histories in a fresh diagnostic namespace if more stage capture is needed.
4. If a genuine implementation defect is demonstrated, isolate its fix and
   rerun required controls under new provenance, retaining this rejection.
   If only the numerical-contract scope issue remains, record that blocker;
   do not refit on these failures or touch HD3 to force acceptance.

No test process is currently needed for the completed QD3 pipeline. Automatic
continuation must do bounded investigation, not poll/restart rejected QD3.

## Original tensor and prefix verification (A3,17:54UTC)

`verify_rejected_captures.py` reloaded202 original captures (88 for math11,
114 for math19 across both modes), checked recorded tensor hashes, finite
logits, same actual inputs/positions and history hashes, terminal no-defer,
and producer/consumer/pending transition order through each failing target.
Candidate logits and the three margins reproduced exactly from raw tensors.
math11 has3 earlier CPU deferred routes (E38 twice,E16 once); math19 has8
(E16 four,E38 once,E41 twice,E50 once). Neither failure is a current-step
deferred event. Evidence: `evidence/qd3-v1-rejected-original-verification.json`.

The first draft forensic script incorrectly counted all nonnegative partition
IDs as CPU routes and raised PRODUCER_PENDING_ROUTE_MISMATCH before producing
a report. Partition IDs can also be NPU-owned; the corrected tool counts only
selected CPU-owned IDs matching their TopK position, as the frozen exporter
already does. This was a new diagnostic-tool error, not a production failure;
the frozen exporter/evaluator/production code were not modified.

This verifies recorded transactions, not independent arithmetic correctness.
The next bounded test is replaying these TWO fixed baseline histories with
the existing explicit-replay oracle versus the production deferred path in
fresh diagnostic directories, retaining the same selected experts and budget.
That can test whether the observed prefix differences follow intended
next-layer injection semantics. It must not be called qualification or change
the rejected candidate decision. Historical Round6.1 replay is supporting
context only; it does not substitute these new history-specific results.
