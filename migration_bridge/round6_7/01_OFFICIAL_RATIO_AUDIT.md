# Official ratio and policy audit — CODE_INSPECTED

Pinned official reference:
[Expert Scheduling Tutorial, 0dce4c9](https://github.com/kvcache-ai/ktransformers/blob/0dce4c9b097710951d1b3e232f50b973ff0a5505/doc/en/kt-kernel/experts-sched-Tutorial.md).
Inspected 2026-09-17. Official CLI treats `kt-num-gpu-experts` as per-layer
count and `kt-gpu-experts-ratio` as total routed-pool fraction; ratio overrides
count. Placement policy and budget are independent controls. Frequency accepts
an explicit initial activation-statistics file; dynamic update is separately
enabled. Official Qwen/NVIDIA throughput is not imported as an Ascend result.

Our experiment deliberately uses uniform nested expert sets and different
resident budgets, without fitting frequency or enabling dynamic. The deployed
loader requires `frequency + init-expert-location` to ingest our explicit
binary counts. This is a **mask loading mechanism**, not a trained frequency
policy; actual physical IDs/counts are audited after load.

All-MoE NPU-resident count: R64 1664, R32 832, R16 416, R8 208.
Theoretical routed BF16 payload only: 26.8125 / 13.40625 / 6.703125 /
3.3515625 GiB. Actual device footprint includes non-routed weights, fixed KV
capacity4096, workspaces and allocator behavior, and must be measured.
