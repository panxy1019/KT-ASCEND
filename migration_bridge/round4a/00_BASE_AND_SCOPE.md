# Round 4A Base and Scope

## Frozen repository state

- Round 3 parent/frozen Round 4A base:
  `f0bfd6b9eb78b33871c9f2d2bb7fb9b06df5d9be`
- Round 3 parent tag: `round3-a3-verified`
- Round 4A parent branch: `feature/kt-multiexpert-multilayer-tp1`
- Round 3 SGLang/frozen Round 4A child base:
  `06b319dc3a62b77c880e36b042d273bfc3957d12`
- Round 4A SGLang child branch: `feature/kt-ep-multiexpert-tp1`

Both repositories were clean when the branches were created. The child SHA
matches the Round 3 summary and submodule pointer.

## Round 3 numerical reference

```text
Layer 17 / Expert 8
CPU repeat: 10/10 byte-identical
CPU vs BF16-rounded FP32 relative L2: 5.29e-5
CPU vs NPU relative L2: 0.004299
Full-model A/B: 15/15 token-ID exact
Maximum observed |delta logprob|: 0.08105409145
```

These values are a reference, not thresholds to be silently relaxed. Round 4A
keeps the expert correctness limits at `5e-4` and `1e-2`, and the full-model
logprob limit at `0.20`.

## Scope boundary

Round 4A changes only placement scale under DeepSeek-V2-Lite, TP=1, BF16 and
graph-off execution. Deferred experts, dynamic placement, MTP, speculative
decoding, quantized NPU weights, TP>1/HCCL, KML, NUMA/threadpool scaling and
performance optimization remain off. The LLAMAFILE 1408 path retains
`threadpool_count=1`.

The Round 3 ACL stream synchronization, synchronous D2H/H2D, owned host tensor
lifetime, CPU route sanitization, int32 CPU mapping and physical NPU expert
count invariants are frozen correctness requirements.

## A3 isolation preflight

- Host: `a3-server-00`, openEuler kernel
  `6.6.0-72.0.0.76.oe2403sp1.aarch64`
- Disposable container: `kt-r3-dsv2lite`
- Container CPU set: `0-15`
- Container memory limit: `137438953472` bytes (128 GiB)
- Assigned accelerator: NPU 0 only
- NPU 0 at preflight: no running process
- Container workload at preflight: `sleep infinity` only
- Host/business containers: not stopped, restarted, reconfigured or modified
