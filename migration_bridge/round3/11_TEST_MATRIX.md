# Round 3 Test Matrix

| Domain | Gate | Result |
|---|---|---:|
| Base | Round 2C parent and SGLang child bases frozen | PASS |
| Source | kvcache-ai repositories and exact model snapshot | PASS |
| All-NPU | load, prefill, decode 1/8/32/64, generation | PASS |
| KT EP | CUDA dependency audit; no mechanical rewrite | PASS |
| KT EP | current stream, event, native handle, submit/sync | PASS |
| KT EP | synthetic wrapper E2E | PASS |
| Weight | same revision and artifact fingerprints | PASS |
| Expert | real CPU, real NPU, numerical equivalence | PASS |
| Layer | CPU-not-hit, mixed-hit, sequential, overlap | PASS |
| Semantics | global IDs; weights/scaling/shared expert once | PASS |
| Prefill | CPU hit and all-NPU top-1/token agreement | PASS |
| Decode | 1/8/16/32/64, CPU hit, exact token IDs | PASS |
| Generation | 16/32/64 and three prompt families | PASS |
| Stability | two 576-token stability campaigns | PASS |
| Memory | no monotonic host or NPU leak | PASS |
| Regression | Round 2A / 2B / 2C | PASS |
| P0 | TP1, graph/deferred/dynamic/MTP/speculative off | PASS |

Final result: `DEEPSEEK_V2_LITE_TP1 = A3_VERIFIED_READY` for the bounded MVP
configuration documented here.

