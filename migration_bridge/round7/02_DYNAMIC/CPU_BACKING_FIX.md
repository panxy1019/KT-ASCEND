# Dynamic CPU backing load defect

Date: 2026-09-17 UTC. Before evidence: A3_VERIFIED. Repair qualification: pending.

The original-binary L1 eviction regression loaded frozen Uniform R8, evicted
predetermined logical experts0/56 through the official in-place wrapper update,
and tested historical capture1. E0 has finite, ten-repeat exact output but
relative-L2=1.0 against the canonical BF16-visible reference, violating the
unchanged0.01 gate. This is a reproducible Dynamic ownership-transition defect,
not evidence that completed Static execution is wrong.

Root cause: multi-subpool LLAMAFILE reshuffles CPU weight slices during load but
uses `should_skip_expert()` with the **initial execution ownership** mask. Thus
initially NPU-owned experts have no valid CPU backing slice after eviction.
Single-pool mmap aliasing does not have this same skipped-copy seam.

Minimal production change removes only this load-time ownership skip. Mapping
permutation/range checks remain; execution-time ownership skips remain. Weight
identity, arithmetic/casts/scaling and frozen masks/CPU64 are unchanged.

The pinned [official LLAMAFILE source](https://github.com/kvcache-ai/ktransformers/blob/0b954fe4ca2f9e79519da66156e6a3f92b686232/kt-kernel/operators/llamafile/moe.hpp)
was rechecked during diagnosis; it does not supply a verified fix for this
specific C64/F32 eviction case. No wholesale upgrade is performed.

Original native SHA:
`a5497fa085d704d8f67484b3936eaf448080f5d445899d88e4e76bea4305f077`.
Original compiled header SHA:
`17a7e720166b18ee6fb57e4de8ffc683972df08ca98a56c7cd763f11c5a642eb`.
Failure archive A3 SHA:
`e9ba345f59f50e78edec94c2d1e77b1a3cea90aa0d1280b82b9f8df51db31a16`.
Failure archive is now preserved locally under `raw/`: SCPexit0, local/A3 SHA
match, and full tar validationexit0. Original failure and binary remain intact.

Minimal fix commit `4bfe9c44f` (resolve full before external reporting), private
build tool commit `b18b72b67bfd83883178e3aec538c9052f9915f1`. Repair patch archive
SHA `be1489cd8efd7c57238fa90616206d4e082d98e00118cb6f8ec1877516d2430b`
was transferred completely and verified on host/container before extraction.
Repaired header SHA:
`73a863bbc5637860f126d8c4e346c4e8cfe53f4f9a7cde8ede5c17583754b382`.
New build runs detached at `/workspace/c64-dev/round7/backing-fix-build-v1`;
`build.json` is BUILDING at inspection, log `backing-fix-build-v1.log`.
No post-fix numerical result exists yet; do not claim the bug closed on build.
All71 existing pure-Python local tool tests pass; these do not exercise native
weight loading and do not substitute for the repaired A3 eviction regression.

`build_full_cpu_backing.py` snapshots the original source/objects/runtime into a
new owned Round7 directory, substitutes only the repaired header, recompiles
bindings and links a **new** native binary. It explicitly verifies original
source/binary immutability. A build PASS alone is not a numerical gate.

Next required verification: same eviction regression L1/L13/L26 under repaired
binary, representative operators, Uniform/Frequency compact, fresh R64 under
common-loader-change rule, updated imported/native provenance. Only then may
full-model Dynamic qualification proceed. No Dynamic threshold is frozen yet.
