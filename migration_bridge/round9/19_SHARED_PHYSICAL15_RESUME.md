# Round9 physical15 shared execution resumed — 2026-09-20

User resolved the host/device ambiguity explicitly: **110.123.0.3 physical15**,
sharing with the existing training. The earlier .22 selection and pause are
superseded. No training container or global device mode was changed.
Delegated review uses **gpt-5.6-sol / medium**, as requested.

## Shared access: new evidence supersedes the earlier blocker

The nonprivileged container exposes only `/dev/davinci15` plus the standard
manager/SVM/HDC devices. `ASCEND_RT_VISIBLE_DEVICES=0`. ACL reports one device;
`aclrtGetPhyDevIdByLogicDevId(0)` returns physical15. Host configuration remains
`UNION_POLICY`, `device-share=False`.

Previously `npu-smi` returned -9005 inside this restricted container. This
management-query failure did not establish ACL compute failure. Actual device
mapping and computation now pass without changing the host modes. This is
empirical support for this exact image/driver/container configuration; it does
not qualify other versions, hard resource isolation, or arbitrary workloads.

Image: `sha256:601e76d17e9fb4b1c5a72d491cf086c533a309c5c651315e73fe783c90460c8a`.
Host driver26.0.rc1, image CANN9.0.0, torch2.9.0 / torch_npu2.9.0.post2.
Runtime probes: CPU quota2, host RAM4GiB, memory-swap4GiB, network none,
nonprivileged, no additional capabilities. CPU quota is not exclusive cores.

| Probe | Result | Driver-reported own process HBM peak |
|---|---|---|
| ACL mapping/init/finalize only | PASS | No tensor allocation |
| 1024-element FP32 arithmetic | PASS, exact CPU comparison | See raw logs |
| 256MiB allocation/write/read/free | PASS | 379MiB |
| 1024MiB allocation/write/read/free | PASS | 1147MiB |
| 2048MiB allocation/write/read/free | PASS | 2171MiB |

Driver samples show training PID2657094 and the owned probe simultaneously on
Card7/chip1. Physical15 total HBM peaked at9292/10060/11084MiB across the three
steps. Before testing8969MiB; after2GiB8970MiB; afterGR8969MiB. Each owned
container exits0 and disappears from the NPU process list; torch allocated and
reserved both return0. Training remains present with6126MiB process memory.
This does not measure training throughput or prove zero slowdown.

The first tiny probe failed: it finalized a manually initialized ACL runtime
then attempted PyTorch initialization in the same process. The revised probe
runs the mapping checker in a child process, leaving PyTorch as sole runtime
owner. Both versions and the failed log are retained. No threshold was changed.

Full-model resource admission remains pending: training future peaks are not
bounded by these observations. The tests used a conservative4GiB runtime
allowance plus12GiB reserve, but this is program-side admission, not hardware
quota enforcement. The ladder monitoring records evidence; it is not a complete
production watchdog and must not be reused as full-model admission.

## GR eager reference and numerical evidence

Added `runtime/hyperconnection.py`, following the mathematical definitions and
parameter names from SGLang commit4ccff141dbe992794f9da6c3aa23535b4f72000d.
Four branches, lowrank320, width2560, per-branch Gemma RMSNorm, gated input mix
and residual injection; eager PyTorch only. MTP excluded. This is an isolated
reference adapter, **not integrated into SGLang's model runner**.

CPU RED: adapter missing. GREEN: four tests cover distinct residual branches,
nonzero gates against independent double equations, empty batch/invalid width,
and split-batch equivalence. General tooling suite:37 tests PASS (includes
recovered explicit host/device admission profile changes from .22).

Physical15 probe uses original checkpoint weights from layers0/23/24/47,
attention and MLP GR; F32/BF16, token counts1/3/32, three deterministic repeats.
48 configurations,144 paired mix/combine executions. Inputs are seeded synthetic
activations, not captured full-model activations. All outputs finite.

| Precision | Frozen relative-L2 gate | Maximum mix error | Maximum combine error |
|---|---|---|---|
| FP32 | 0.0001 | 6.418827199853731e-7 | 1.1527427962200615e-7 |
| BF16 | 0.02 | 0.004077069507414718 | 0.0029889418319903236 |

Reference is independent CPU double equations with identical original BF16
stored weights. The BF16 comparison includes BF16 compute rounding. NPU HF32
matmul disabled. Peak torch allocation53858816bytes; allocated/reserved0 after
cleanup. These thresholds qualify only this local GR probe, not model quality.

## Source continuity and remaining gates

Destination identity receipt SHA256 rechecked:
`7d6cb3db19497303bdd8ceb24efae626e1926a610ab7fbd4406b47d05db4a239`.
No redundant full model transfer or360GB hash pass.

Exact missing SGLang pin f02f304a25364d1e8fe9c52515a22f20f5089a52 was recovered
from the clean local round6 submodule. The bundle is now on .3 and independently
cloned at `/mnt/data1/kt-round9/src/sglang-f02-recovered`; no gitlink substituted.
Bundle SHA2569be816ac5f1685b57edb6bf6de3d0cb571b47e7e2b6ca08c909f273e52dc8979.
The .22 uncommitted source was archived before reuse; original files remain.
Archive SHA2565ed958bf59a39e9966ffc286956af4295f838d018e9c79e0b248ede07d93a40d.
Host-specific admission changes and GR tests recovered; the NUMA-argument and
.22 resource-probe changes remain in the preserved archive, not silently lost.

Current outcome: shared bounded runtime PASS, prior bounded BF16/C64 CPU evidence
retained, local GR PASS. QSA/GDN/PLE, model registration/integration, complete
operator/ownership gates, full-model reference, CMMLU, lowbit, C128, asynchronous
PLE/scheduling/context/performance remain unfinished. No stage-wide PASS inferred
from GR alone. Next: source semantics and synchronous module gates before any
full-model load. No round7 automation, Round8 performance, push, merge or tag.

Raw evidence: `evidence/physical15/20260920-shared-resume/`; scripts, failed and
successful logs, driver samples and container states retained with SHA256SUMS.

## Focused review and corrections

Fresh-context review by gpt-5.6-sol / medium found one important evidence issue:
the GR probe initially inferred revision from directory name. Fixed by requiring
the frozen full-verification receipt SHA256, revision and manifest identity,
exact approved model path, live sizes for all144 files, and live SHA256 checks
for config/index. A negative test rejects wrong receipts/revisions/manifests,
wrong directories, changed same-size metadata and missing files. RED missing
helper; final suite37/37 GREEN. This binding reuses the trusted prior full hash
receipt, and does not claim a fresh hash of every checkpoint byte.

Minor test gap retained: the adapter's optional hc_per_branch_norm=False branch
and isolated CPU BF16 cases are not covered by the four CPU tests. The actual
model's per-branch BF16 path is covered by the NPU probe; no other mode inherits
that qualification.

During the identity-bound rerun, fresh host telemetry showed HBM4% and all
original training PIDs absent from the NPU process list. Their exit reason was
not investigated, and no training stop/restart/change was issued. The earlier
ladder and first GR run establish the shared-window evidence; the identity-bound
rerun must not be presented as an additional simultaneous-training run.

Identity-bound rerun PASS:48 configurations,144 repeats,144 live file sizes
checked, same frozen error gates, peak allocated53858816bytes, cleanup0/0.
Both initial and corrected results retained; no source or math change occurred.
