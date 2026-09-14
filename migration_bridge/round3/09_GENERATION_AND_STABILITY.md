# Generation and Stability

## Generation

- Greedy generation 16: PASS
- Greedy generation 32: PASS
- Greedy generation 64: PASS
- English/Chinese/structured-numeric: PASS
- Hybrid versus all-NPU token IDs: identical in all 15 matrix cases

## Stability

Initial final-placement run:

- 3 cycles × 3 prompts × 64 tokens = 576 generated tokens
- 9/9 requests exact to baseline
- all logprobs finite
- container memory before/after: about 55.7 GiB, no monotonic host leak
- NPU process memory: 30,548 to 30,610 MiB (+62 MiB cache high-water)

Post-race-fix confirmation:

- another 3 × 3 × 64 = 576 tokens
- 9/9 exact to the saved Hybrid E8 baseline
- all logprobs finite
- NPU process memory approximately 30,584 MiB
- container memory approximately 55.83 GiB
- no ERROR or Traceback in server log
- server terminated cleanly and NPU 0 returned to no-process state

The measured Hybrid/all-NPU latency ratio after first-request warmup was generally
1.58-1.89×. This round establishes correctness and stability, not performance parity.

