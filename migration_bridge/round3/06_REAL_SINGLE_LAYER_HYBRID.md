# Real Single-Layer Hybrid

## Placement and semantics

- Real model layer: 17
- Final CPU expert: global ID 8
- NPU experts at that layer: 63
- Router owner: SGLang DeepSeek layer
- Routing weights: applied once
- `routed_scaling_factor`: existing DeepSeek location, once
- Shared expert: existing DeepSeek location, once
- Deferred experts: disabled

## Cases

The real layer verifier covers CPU-not-hit, CPU-hit/mixed routing, CPU/NPU expert
identity, sequential Hybrid, and overlapped Hybrid. Global IDs are retained on
the CPU side; the NPU side uses a dense physical mapping and disabled CPU routes.

Results:

- CPU-not-hit: PASS
- Mixed CPU/NPU hit: PASS
- Sequential Hybrid: PASS
- Overlapped Hybrid: PASS
- Sequential versus overlap: exact for the synthetic bridge and within the
  documented BF16 tolerance for the real expert
- NaN/Inf: none

Tools: `verify_real_expert_identity.py` and
`verify_real_single_layer_hybrid.py` in this directory's `tools/` folder.

