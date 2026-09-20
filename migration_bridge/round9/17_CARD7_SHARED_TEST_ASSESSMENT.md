# Card7 shared testing assessment — 2026-09-20

User requests consideration of sharing card7 on110.123.0.3 with existing
training. This permits feasibility assessment; it is not evidence of admission.
All operations in this assessment were read-only. No NPU context/model was
started, no device mode changed, and no training process/container modified.

## Destination data gate now verified

Destination state is REPLICATION_IDENTITY_PASS; final revision directory exists.
Destination receipt verification-1789886190714568317.json SHA256:
7d6cb3db19497303bdd8ceb24efae626e1926a610ab7fbd4406b47d05db4a239.
It reports PASS, revisionde4b8e4d43b917e7706784d8bb445c9af86a3540,
manifestSHA62e6eca0f2900745dbfd9ba6ed76e60547e0d602acf2d62a9ecac3972028b38e,
144files,131shards,1658tensors and359999963128payloadbytes.
The receipt's144PASS file names/sizes match the frozen manifest and current
published directory sizes. No redundant full360GB hashing run was launched.
Model execution remainsNOT_RUN. Main/source semantic and runtime gates remain.

## Read-only sharing evidence

At07:18–07:20UTC, card7 has chip0/physical14 and chip1/physical15, each65536MiB
HBM. chip0 usage14%,chip1 usage13% (rounded npu-smi telemetry). This suggests
roughly55GiB instantaneous HBM free per die, not a reserved capacity allocation.

Three observations at07:20:10,07:20:22,07:20:33UTC:

| Die | AICore usage | NPU utilization | HBM usage |
|---|---|---|---|
| physical14 |22%,60%,23%|41%,60%,37%|14% all three|
| physical15 |28%,12%,41%|48%,34%,44%|13% all three|

This short sample does not establish a long-run peak or prove15 is consistently
less loaded. Physical15 is a tentative first bounded-probe target because its
observed memory footprint is slightly smaller, not because it is idle.

- device-share=False for both chip0 and chip1 (also checked asroot).
- chip1 vnpu-mode=docker; info-vnpu lists0virtual instances. Its static resource
  free/total table is not runtime free HBM: training allocations still exist.
- physical15 process2657094 is python3.10,6126MiB reported process memory.
  Its cgroup belongs to mzy-train container0419bcfee46f…; physical14's known
  process2657093 also remains running. Training processes have run~4h24m.
- mzy-train is privileged, exposes davinci0–15, has no explicit Docker cpuset
  or host-memory limit. No claim that it exposes vCANN resource quotas.

## Decision and scope

Hardware capacity makes shared correctness development plausible. Current
configuration does not establish an isolated second-container shared runtime:
container sharing is disabled and no vNPU is present. This is not proof that
all same-host/same-container multiprocess use is impossible. Do not enter or
alter the training container as a shortcut, and do not assume a new container
has hard HBM/compute quotas merely because its visible device is restricted.

Proposed progression, once a supported access/isolation route is established:

1. Keep CPU-only source/loader/format work running independently.
2. One owned process on physical15, tiny allocations then bounded operator
   work. Engineering budget2–4GiB HBM, no full model or concurrency. Monitor
   aggregate and owned memory; reclaim by terminating only the owned process.
   This budget is not an existing driver-enforced partition or a guarantee of
   zero effect on training.
3. If co-existence and release are verified, extend real-operand correctness
   probes within a separately reviewed budget (e.g.8GiB), single-request/small
   prefill. Record all evidence as SHARED_DEVICE; preserve timeout failures.
4. FullQ64 requires its own measured peak budget and training peak headroom.
   28.125GiB resident expert payload alone is insufficient to certify fit; shared
   experts/attention/GR/runtime/KV/loading copies remain. No admission yet.
5. Formal performance, C64/C128 speed comparison and long-context capacity
   limits still need a controlled resource window. Shared runs can provide
   correctness evidence but do not establish isolated throughput or latency.

Resource contention may affect an entire distributed training job if this rank
slows; topology/process count alone does not prove its communication pattern.
Current user request does not authorize stopping/restarting the training task.

## Primary documentation boundary

Official soft-sharing practice:
https://www.hiascend.com/developer/techArticles/20260310-1
describes time-sliced compute and memory quotas through vCANN-RT, requires
device sharing, restricts that documented setup to nonprivileged containers,
and requires idle AI workloads for setup. It lists A2 hardware, so this A3
machine/driver26.0.rc1 needs its own support check. Do not install/reconfigure
that solution against the active privileged training container.

Official A3 npu-smi overview:
https://www.hiascend.com/doc_center/source/zh/HDK/2610/A3/A3npu/npusmia3_0003.html
distinguishes virtual instances from physical-device use. No actual virtual
partition was created here. Device sharing and resource isolation are separate
questions; both must be understood before changing this machine's mode.
