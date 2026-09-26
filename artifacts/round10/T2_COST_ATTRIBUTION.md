# T2 Hybrid cost attribution and replay fixture

Status: `PASS_T2_HYBRID_COST_AND_FIXTURE`. No extra model start was needed.

The v95 host-only observer recorded 39 real Hybrid scheduler chunks and 78 complete sampled layer applications at layers 0 and 3. For 32-token chunks, the enclosing apply median was 3.430 ms, CPU result wait 1.311 ms, CPU submit return 0.302 ms, and resident NPU apply return 0.394 ms. These are nested host-clock intervals. They are not added together or multiplied by layer count. The NPU interval is a launch/return duration without a forced device synchronization.

The exposed CPU result wait is measurable but is not the sole sampled Hybrid cost. Layer0 also contains one cold-start outlier (868.612 ms) while its median is 3.614 ms; this is kept in the evidence and is not treated as steady-state speed. Request-level evidence covers H64 and natural Decode (32 tokens to length and 49 tokens to EOS), while per-stage device timing for Decode remains unsynchronized and is not claimed.

A replay fixture was frozen from the qualified v80 real request capture: two adjacent layer0 groups with real hidden states, TopK weights/IDs, and independent BF16 contributions. The tensor hash is `c837641468b9db93c422a3610af7a7ecc1f14d4bbb359841d69ac0dfd655dc6e` and the complete 724-group route tape hash is `201a9b65b24c2fd9774c8c2dbfc4f795078d12dc8f20ce4b2a82efee12d857c0`. The old capture itself is diagnostic, not performance evidence; Round10 uses it as immutable T3 numerical input.
