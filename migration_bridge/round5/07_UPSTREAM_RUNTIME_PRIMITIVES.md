# Round 5 — Raw Upstream Runtime Primitives

Evidence state: `A3_VERIFIED`.

## U1 — callback worker lifecycle

The raw extension exported:

```text
init_ascend_callback_worker
subscribe_ascend_stream
shutdown_ascend_callback_worker
is_ascend_callback_worker_running
```

On NPU0, the observed state transition was:

```text
not running
-> init: running
-> subscribe current native stream: running
-> shutdown: not running
```

The stream handle was non-zero. Result: `PASS`.

## U2/U4 — framework pinned D2H/H2D

The committed probe copied a 4096-element FP32 tensor NPU→pinned CPU→NPU for
1000 iterations. Every D2H and H2D result had the same SHA256 as the source:

```text
c7c0a32d5f43b1b6ec256a55fc5c1bf2d789a5a28d188cd3b69f50866dc16482
```

| Metric | Result |
|---|---:|
| D2H unique hashes | 1 |
| H2D unique hashes | 1 |
| source/hash equality | exact |
| finite | PASS |
| status | PASS |

## U3 — CPUInfer immediate submit and host drain

A deterministic synthetic LLAMAFILE MoE used one NUMA subpool with eight
workers, four experts, top-k two, hidden 256 and intermediate 256. It performed
1000 identical `CPUInfer.submit` → `CPUInfer.sync` forwards.

| Metric | Result |
|---|---:|
| forwards | 1000 |
| unique BF16 raw-bit hashes | 1 |
| output hash | `0035e4f3593895a0b165205eb06672b9c3b8bced7043fc043cfc4ef309052c67` |
| all finite | yes |
| status | PASS |

Two probe-development failures are retained for audit:

1. v1 accidentally created eight subpools for a one-block synthetic shape;
   the kernel correctly rejected it.
2. v2 executed the kernel but the tool attempted NumPy conversion of BF16;
   the tool was corrected to hash the raw `uint16` bit representation.

Neither is classified as a runtime failure. v3 is the qualified run.

## U5 — overlap status

Static inspection confirms upstream's NPU immediate path performs host submit
to WorkerPool and then allows independent NPU experts to run while the pool is
active. The primitive probes do not measure overlap duration. Therefore:

```text
U5_OVERLAP_ARCHITECTURE = CODE_INSPECTED
U5_OVERLAP_RUNTIME      = PENDING_R2C_AND_P1
```

## Static lifecycle risk discovered

The pinned upstream `CPUInfer::sync()` heap-allocates `SyncArgs` and the direct
host path does not release it. Callback submission also lacks the local
exception/RAII wrapper. The small leak is not visible in the functional
1000-forward hash test, but the code differs from the Round 4A lifecycle fix.
It is a minimal alignment candidate and must be covered by lifecycle/RSS tests.

```text
R5_G2_UPSTREAM_RUNTIME_PRIMITIVES = PASS_WITH_ALIGNMENT_RISK
```

## Evidence

- `evidence/a3_raw_runtime_primitives.json`
- `evidence/a3_raw_cpuinfer_llamafile_submit_v1.json`
- `evidence/a3_raw_cpuinfer_llamafile_submit_v2.json`
- `evidence/a3_raw_cpuinfer_llamafile_submit_v3.json`

