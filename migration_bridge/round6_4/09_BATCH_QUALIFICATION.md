# Batch2/4 engineering qualification — in progress

No batch quality or speedup approval has been issued. Batch1 remains the sole
primary quality endpoint. Deferred OFF and ON use distinct same-path references.

## Attempt v1: tooling failure before requests

`r64-batch-campaign-v1` fails on its first batch2 OFF model load:
`NameError: generate_gpu_experts_masks is not defined`. No request ran.
The KTEP import guard caught an ImportError, leaving its imported symbols absent.

Root cause is the new test-only capacity adapter's `execv` argv[0]. It passed
the virtualenv executable but the bare program name `python`. CPython discovered
the system prefix, not the virtualenv containing kt_kernel. Read-only A3 A/B:

| argv[0] | sys.prefix | kt_kernel discovery |
|---|---|---|
| python | /usr/local/python3.11.15 | None |
| /workspace/results/venv-r6/bin/python | /workspace/results/venv-r6 | installed package |

The corrected adapter preserves the full interpreter path as argv[0]. It still
changes only capacity/batch CLI flags. No production model, routing, placement,
kernel, or Deferred code changed. A local behavioral exec regression prevents
reintroducing the interpreter identity defect; three adapter tests pass.

The serving guard terminated only its owned failed process group. Its readiness
helper was still waiting for a server that had exited; that exact owned helper
was then stopped. The wrapper now observes the guard while waiting and exits
immediately on early server death. Three owned-process lifecycle tests pass.

Failure evidence is retained in `evidence/r64-batch-campaign-v1` including the
resolved argv, actual metadata, traceback, and guard termination report.

## Attempt v2

New root `/workspace/results/r64-batch-campaign-v2` starts the unchanged frozen
16-request short test. Sequence: batch2 OFF/ON, batch4 OFF/ON, then ten repeats
and two independent restarts per profile, each followed by full transaction
audit. Never reuse the v1 failed result as a PASS. Actual model batch, RID/KV
association, protected-expert union, route partition, terminal drain, and
repeat schedule must all be verified before reporting engineering success.

### Short-test results (A3_VERIFIED)

| Profile | Requests | Physical forwards | Actual model batch | Deferred routes | Transaction audit |
|---|---:|---:|---:|---:|---|
| batch2 OFF | 16 | 520 | all2 | 0 | PASS |
| batch2 ON | 16 | 520 | all2 | 85 | PASS |
| batch4 OFF | 16 | 260 | all4 | 0 | PASS |
| batch4 ON | 16 | 260 | all4 | 44 | PASS |

All physical forwards reconcile to known RID histories/KV positions; route
partition, global protected union, terminal drain, tensor hashes and finiteness
pass. Batch2 ON has83 exposed forwards; batch4 ON has43. These counts include
the entire captured session and are not a per-token exposure percentage.
The batch2 OFF/ON schedule hashes agree, as do batch4 OFF/ON schedule hashes.
Cross-mode token equality is not required and is not claimed.

The ten-repeat/restart campaign is still running. Short-test PASS alone does
not promote batch2/4 to deterministic, quality-qualified, or speedup-verified.
Archived JSON evidence: `evidence/r64-batch-campaign-v2/*-pilot/`.

### Batch2 OFF repeated/restart result (A3_VERIFIED)

Ten repeats completed160 requests and5200 physical forwards. All5200 actual
batches contain2 rows; all10 normalized schedule hashes equal
`6ab8ba743160654ec4a22eea9263f496e08217211f3f4061d53df82e9a438d42`.
7130 actual CPU tasks and10400 worker completions reconcile with the capture.
Both independent restarts completed16 requests/520 forwards each and matched
the prior process's per-request trajectories and normalized execution schedule.
All three final transaction audits PASS, finite outputs, no unexplained forwards.
The corresponding five JSON evidence files per stage are archived locally.

Deferred ON and batch4 long-repeat qualification remain in progress. No
cross-profile or quality conclusion is inferred from this completed OFF gate.

## Batch2 closed; batch4 first failure

Batch2 ON ten-repeat and both restarts PASS. Ten-repeat:5200 forwards,
7900 actual CPU tasks,850 deferred routes,10400 worker completions; each
restart reproduces16 trajectories and the same normalized cohort schedule.
Batch2 engineering determinism is verified; quality and speedup remain separate.

Batch4 OFF failed on repeat1/cohort0/s6_en_01 at token55: reference7230,
actual11765. Controller stopped this profile before its later stages and
batch4 ON long repeats. It did not continue a failed qualification.

Read-only partial-run audit reconstructs every received response from raw
requests, including the three rows received with the failing row but not yet
added to the early-exit summary. All20 received requests and325 forwards
reconcile. Actual batch is4 throughout, no unknown/incomplete forward,
route partition and terminal checks PASS, no deferred routes, all finite.
The two cohort0 schedule hashes are identical:
`0e18be3d540d2fe551ae78e9e4333a342304a63ae1e3eb1dda665cbc195baab7`.

Earliest difference in retained tensors: router logits/weights and final
logits at decode position1; CPU output first differs at position2. At token55,
the candidate logits change from[27.125,27.0] to[27.0,27.125]. Compact capture
does NOT retain all hidden/attention inputs, so this is not proof of a specific
GEMM/reduction or CPU-transfer root cause. OFF failure cannot be excused as
approved cross-mode Deferred approximation.

Full failure capture(547MiB on A3) is preserved; tensor bytes copied locally
and excluded from Git, with index/hashes and JSON/log evidence retained.
Comparison SHA: `db5f1f03bdc4d3676af6d402cd9880533a31c680872d8fc3ed20e98e7a67f7f2`.
Status: `BATCH4_SAME_PATH_BLOCKED_PENDING_ATTRIBUTION`.
Batch1 remains the originally preregistered primary profile, not a replacement
chosen after this failure. No batch4 release claim is allowed.

The pinned SGLang deterministic-inference document lists FlashInfer/FA3/Triton,
not Ascend. Local document SHA
`5251f06ae93450f60a714982d40fc781da61dbfed64c5c7659365b6e28338c06`;
child `f02f304a25364d1e8fe9c52515a22f20f5089a52`. The general upstream
[deterministic-inference roadmap](https://github.com/sgl-project/sglang/issues/10278)
is background, not evidence of an applicable drop-in Ascend fix. Do not
enable unrelated CUDA batch-invariant kernels or change the frozen runtime.

Bounded diagnostic `r64-b4-off-diagnostic-v1` repeats the same4 cohorts twice,
adds full snapshots at0,1,55,260,261,315, and otherwise preserves configuration.
It is not a substitute for the failed ten-repeat gate and cannot qualify batch4.

### Bounded diagnosis result

The full-snapshot repetition reproduces exactly the same token55 flip and
passes the same325-forward integrity audit. Full snapshots show no common
tensor differences at prefill0 versus260. At decode1 versus261, Layer7
decoder_input_hidden and decoder_input_residual already differ, before
Layer7 attention/router/CPU expert execution. Layers before7 are NPU-owned
in the frozen D1 placement. Thus the earliest observed decode difference is
upstream of that step's first CPU expert. This narrows investigation to
pre-Layer7 NPU/KV execution; it does not identify an exact operation or prove
all prior KV contents identical. No unsafe CPU-transfer fix is justified by
these results.

`evidence/evidence-b4-diagnostic/` preserves the complete index, response,
comparison, metadata and guard/reaping evidence. Full tensor snapshots remain
at `/workspace/results/r64-b4-off-diagnostic-v1` on A3.

Round6.4 batch4 disposition: **EXPERIMENTAL_NOT_QUALIFIED**,
`SAME_PATH_NONDETERMINISM_PRE_L7_NPU_OR_KV`, no batch4 quality/speedup release.
Next minimal separate investigation is an all-NPU batch4 control with captured
per-layer inputs/KV metadata, followed by the first differing NPU operator;
do not change workers, placement, or arithmetic without that evidence.

The originally frozen sole primary endpoint is D1 batch1. Its full same-process,
restart, oracle and system tests passed independently before batch expansion.
Continue its approved quality/performance/stability work; this neither lowers
batch4's exactness gate nor promotes its failed experiment. Batch2 also remains
engineering-only until its own quality protocol is exercised.
