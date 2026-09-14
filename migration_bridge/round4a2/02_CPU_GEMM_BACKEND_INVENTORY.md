# CPU GEMM Backend Inventory

| Backend | Source/version | ARM | Buildable in container | Operand contract | Accumulation | Decision |
|---|---|---:|---:|---|---|---|
| LLAMAFILE SGEMM | current kt-kernel | yes | yes | F32 materialized BF16-valued X/W | F32 | frozen baseline |
| repository KML path | stale references only | nominal | no | missing current source directories | unknown | rejected: `BACKEND_BUILD` |
| repository BLIS path | AMD/AOCL INT8 path | no for this use | no for F32 expert | quantized | INT32 | rejected: dtype/architecture mismatch |
| OpenBLAS pthread | Ubuntu `0.3.20+ds-1` | yes | yes | F32 | F32 | isolated winner; full P1 failed |
| BLIS OpenMP | Ubuntu `0.8.1-2` | yes | yes | F32 | F32 | rejected at numerical/runtime gate |
| ATLAS | Ubuntu `3.10.3-12ubuntu1` | yes | yes | F32 | F32 | rejected at numerical gate |
| Arm Compute Library NEGEMM | Ubuntu `20.08+dfsg-5` | yes | yes | F32 via probe shim | F32 | rejected at numerical gate |
| Arm Performance Libraries | proprietary install unavailable | yes | no reproducible pinned artifact | F32 | unknown | rejected before experiment |

KML was not assumed to be the answer. The current tree references
`operators/kml` and `operators/moe_kernel/mat_kernel/kml_kernel`, but neither
source path exists in the frozen repository. Restoring historical proprietary
code would be a new unsupported backend port, not a Round 4A.2 candidate build.

Candidate manifests are under `backend_manifests/`.
