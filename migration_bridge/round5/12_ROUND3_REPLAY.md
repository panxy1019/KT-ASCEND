# Round 5 — DeepSeek-V2-Lite Round 3 Replay

Evidence state: `A3_VERIFIED`

## Frozen profile

```text
model: deepseek-ai/DeepSeek-V2-Lite
revision: 604d5664dddd88a0433dbae533b7fe9472482de0
TP: 1
device: NPU0 only
CPU layer: 17
CPU expert anchor: 8
CPU backend: LLAMAFILE, F32 GGUF with BF16-valued weights
NPU backend: Ascend BF16/unquantized
Graph/Deferred/Dynamic/MTP/Speculative: OFF
```

## Serving replay

The aligned full model loaded successfully and completed both prefill and
decode.  A 16-token greedy request returned finite logprobs in 2.86 seconds.
Ten repeated requests in the same serving process produced one exact token
trajectory; all logprobs were finite.

## Strict Layer17 replay

The production KTEP path was captured with a route that hits CPU Expert 8.
Results:

| Invariant | Result |
|---|---:|
| CPU route hits | 2 |
| distinct CPU/main streams and staging | PASS |
| KTEP vs explicit CPU+NPU reference relative-L2 | 0 |
| KTEP vs explicit CPU+NPU max error | 0 |
| CPU-not-hit control | exact |
| mixed Hybrid vs all-NPU relative-L2 | 0.0032707124 |
| all outputs finite | PASS |

The mixed-backend difference is inside the frozen expert safety gate
`relative_l2 <= 1e-2`; CPU/NPU bitwise equality is not claimed.

## Tool correction

The historical replay tool called the pre-alignment wrapper without a stream
argument.  It was updated to pass the active framework stream handle.  This
was a stale instrumentation interface, not a production failure; the corrected
tool then passed.

## Gate

```text
R5_G6_ROUND3_REAL_MODEL = A3_VERIFIED_READY
```

Evidence:

- `a3_aligned_round3_generate_v2.json`
- `a3_aligned_round3_repeat10.jsonl`
- `a3_aligned_round3_repeat10_summary.json`
- `a3_aligned_round3_real_layer17_v2.json`
- `a3_aligned_round3_real_layer17_v2.log`

