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

Full QD defer=0 capture is running as 16 × 64 matched-history positions. The
HD corpus remains reserved and has not been used for fitting or validation.
