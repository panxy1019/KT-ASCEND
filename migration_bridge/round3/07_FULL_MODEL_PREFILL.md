# Full-Model Prefill Verification

## Final placement

Exactly one routed expert is on CPU: layer 17, global expert 8. The placement
contains 1 CPU slot and 1663 NPU routed-expert slots. Expert 8 was selected from
the all-NPU router trace because it is exercised by the matrix but has only three
hits, avoiding the high-frequency expert-6 perturbation seen during diagnosis.

## A/B matrix

Prompts: English, Chinese, structured numeric. Generation budgets: 1, 8, 16,
32, and 64 tokens. All 15 Hybrid cases produced token IDs identical to all-NPU.
Both sides were internally prefix-stable.

- CPU expert-8 hit count: 3
- Layer-17 routed slots observed: 2,898
- Global routed slots observed: 75,348
- Recorded layer-17 forward passes: 378
- NPU expert count at every recorded layer-17 pass: 63
- CPU expert ID at layer 17: always 8
- Maximum observed logprob absolute difference: `0.08105409145`
- Token divergence: none

Evidence:

- `full-model-all-npu-router-matrix.json`
- `full-model-hybrid-e8-matrix.json`
- `full-model-ab-e8-summary.json`
- `full-model-hybrid-e8-cpu-hit.txt`

