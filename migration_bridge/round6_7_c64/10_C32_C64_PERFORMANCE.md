# C32 / C64 performance protocol

Status: **NOT_RUN — predecessor gates and exclusive host admission required**.

Frozen non-quality prompt manifest SHA256:
`b75565905a7c5065aa05bc1847821f576d0cc0c2018ddf8b05dd2776ca0b0789`.
Input lengths 256/384/512/640/768; greedy one-token output; batch1; radix cache OFF;
same R16 placement, model, tokenizer, GGUF, NPU2 and compiler/native candidate.

Three ABBA blocks, A=C32 (NUMA4, CPU128–159), B=C64 (NUMA4/5, CPU128–191).
Each process warms up on the frozen longest input before five measured shapes.
Primary statistic is the median of block/shape paired C64/C32 wall-latency ratios.
Ratio ≤0.85 is beneficial; (0.85,0.90] marginal; >0.90 no meaningful speedup.
Bootstrap over paired shapes is descriptive, not independent-host replication.

The read-only host admission lease refuses formal timing while another model/controller
or NPU workload exists, or while host CPU/memory admission fails. It never stops R16,
R8, or business workloads. The independent C64 job waits for natural completion.
If isolation is lost during measurement, results fail rather than being labeled formal.

## Attribution boundary

This compares deployable execution profiles, **not a pure core-count scaling test**:

- C32 retains the historical single-TP, unaligned-1408 forward_one path and BF16-visible down boundary.
- C64 uses 704+704 F32 partitions, grouped tail-aware forward_many for prefill and FP32 TP down partials.

Any observed gain combines kernel-path, memory-locality and parallelism changes.
It must not be described as proof of linear 32→64 core scaling. A future optimized
C32 grouped-prefill profile would require its own qualification.

Primary timings have capture and phase instrumentation OFF. Process affinity, anon/file
RSS, NUMA page counts, faults and cgroup usage are recorded outside timed requests.
Unmeasured host bandwidth in GB/s is reported as unavailable, not inferred from CPU usage.
