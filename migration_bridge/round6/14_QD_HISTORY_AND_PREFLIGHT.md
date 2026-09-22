# QD defer=0 history and instrumentation preflight

State: `A3_VERIFIED` for the QD history freeze and one-position preflight;
the complete matched-history numerical gate is **not yet qualified**.

In disposable A3 container `kt-r6-deferred-ascend`, the D1 placement was run
with `--kt-max-deferred-experts-per-token 0`, production overlap, NPU0 only,
and the frozen QD corpus (`fe28d7fe4b4421b53901950b41a7d98b58650de359bd5141e9d05ef71e6bf6f1`).
The generated baseline history contains all 16 QD prompts with exactly 64
greedy tokens each, no failed request, and the expected corpus SHA. It is the
single history source for both forthcoming defer=0 and defer=1 sweeps.

| Artifact | SHA256 (file bytes) | Canonical payload SHA |
| --- | --- | --- |
| `evidence/qd/defer0_history.json` | `1826d8de9895cb2b47b48301813a217adce2ce70e0ec690bf10bc56df558fd68` | `c3042ea828a406b1ccbe42bc70f3652828656cb730b3f75c2b2397388f56b793` |
| `evidence/qd/defer0_preflight_manifest.json` | `5083f7ee50d55cc167c1ec8fe219920d127248c7866a2e028149a90718f1f161` | See `sha256` in manifest |

The preflight restarted the defer=0 service with opt-in tensor dump and
Deferred route JSONL trace. For `qd_en_01`, position 0, the captured logit
argmax matched the serving token. The trace records were isolated to that
request, the final pending set was empty, and route telemetry state was
`POSITION_ALIGNED` (12 trace records). CPU hit count happened to be zero for
this one position; this is an instrumentation check, not a CPU coverage claim.

The full QD defer=0 capture then completed 16 × 64 = **1024** matched-history
positions. All 16 prompt IDs and 64 positions per prompt were present, with
the exact QD corpus SHA and baseline-history SHA. Every request's route trace
was position-aligned and ended with an empty pending set. There were 430
positions with a CPU-owned route. This is a baseline capture, **not** a
defer=1 numerical PASS.

| Full baseline artifact | SHA256 (file bytes) |
| --- | --- |
| `evidence/qd/defer0_manifest.json` | `3f1754ac2a738eb63cba3a8eddfa290ff725eec6053255bb4692a0841060b075` |
| `evidence/qd/defer0_logit_sha256.txt` | `1787a9e04ab705077b8c3c435644f8be7e19e3c197f6d1a471dfb80d2c801a5c` |

The 16 full-logit `.pt` tensors and raw debug traces remain under
`/workspace/results/qd-nodefer-full/` and
`/workspace/results/qd-nodefer-sweep/` in the A3 disposable container.
The hash list identifies the tensor files without committing large raw
arrays to Git. The HD corpus remains reserved and has not been used for
fitting or validation.
