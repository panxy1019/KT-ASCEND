# Round10 A/B coverage delta

| Area | Round10 evidence | Final disposition |
|---|---|---|
| Hybrid policy/quality | ≤128-token input, ACK/EOS, five T1 canary classes, 32/49/64-step decode evidence, two T1 cancel/reuse cycles and final v107 cycle | Narrow qualified; 129/130 live boundary remains unqualified |
| Hybrid cost | Host-only attribution plus replayable real fixture | Evidence frozen; profiling excluded from final E2E |
| Kunpeng Expert | Current native C128/MXFP4 worker path verified against independent same-precision reference; BF16 cross-precision boundary recorded | Current backend retained; no candidate |
| Dynamic | Real Hybrid lifecycle, four commits preserving live state, cancel/reuse | Correct but 4.558176× H0; default OFF |
| GDN | Real weights, layers 0/24/46, FP32/BF16, 1/3/17 tokens, decode/reset state | PASS |
| QSA | Real weights, layers 3/23/47, lengths 1/3/4/9/17, index/compressed-key and signed Top-K checks | PASS |
| PLE | Physical12 original-row exactness, ownership, cancel late completion, reuse and zero credits | PASS |
| Final integration | v107: 8 generations, 1 abort, exact matching outputs, resource/cleanup proof | PASS with reconciled post-cleanup controller filename defect |

Still outside qualification: live 129/130 Layerwise boundary, true long context, multi-request concurrency, HiCache/KV offload, FP8 KV, Graph, MTP, TP2, deployment and broad product quality.
