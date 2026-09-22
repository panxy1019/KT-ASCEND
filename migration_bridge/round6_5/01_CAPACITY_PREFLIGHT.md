# Target capacity preflight

Live read-only host observations 2026-09-16 11:32UTC:

- RAM1TiB, MemAvailable approximately987GiB; swap8GiB, used0.
-256physical cores,4sockets,8NUMA nodes,32cores/node.
-NUMA0 cores0–31; node0 total131767124KiB, free67212928KiB at snapshot.
-NPU0 PCI0000:c1:00.0 sysfs numa_node=6; worker node0 is not NPU-local.
-NPU0 healthy/no process, HBM3411/65536MiB baseline. No other NPU process at
  this instant; not a reservation or exclusivity promise.
-Disk approximately3.0TiB available. Host numactl absent; no host installation.

Decision: retain32workers, avoid conflating cross-NUMA change with ratio effects.
New dedicated container capped192GiB, no additional swap allowance. This is
below current host availability; monitor cgroup usage and host safety reserve.
Memory nodes remain inherited (not a new memory-binding optimization).

Config confirms MoE layers1–26,64experts,TopK6,H2048,I1408.
Expected4992source routed tensors all present in safetensors index.
All-layer logical F32 bytes57579405312 (53.625GiB), not total working-set estimate.
Existing main GGUF covers layers5–12 only (17716742368bytes), hence cannot be
used for all-layer qualification. Independent all-layer export started with
retained exporter under7200s owned-process guard, same fixed model revision.

Full preflight raw JSON on target: /workspace/results/r65-preflight-v1/preflight.json.
Export result and identity verification pending; no claim of completed WP1 yet.
# Runtime locality caveat

The container emits `set_mempolicy: Operation not permitted`. CPU affinity0–31
is verified, but requested strict memory policy must not be labeled effective.
Keep the existing container security configuration during the controlled sweep;
use actual numa_maps/node page counts for placement evidence. Linux distinguishes
CPU scheduling affinity and memory allocation policy. Container runtimes can gate
NUMA syscalls through seccomp/capabilities; the warning alone does not identify
every policy source or prove a bandwidth bottleneck.

Primary references consulted:
- https://cdn.kernel.org/doc/html/latest/admin-guide/mm/numa_memory_policy.html
- https://github.com/containerd/containerd/issues/13610
