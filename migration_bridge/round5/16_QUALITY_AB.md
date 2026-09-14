# Round 5 — Frozen Downstream Quality A/B

Evidence state: `QUALITY_VERIFIED`

The quality input and evaluator were frozen before Round 5 aligned results:

```text
manifest SHA256:
c7bcb09c72f4a7213d0ecdb080f8e88e983ab1a491d4304f448bce28d8f7e1ce

C-Eval: 128 examples
GSM8K multiple choice: 128 examples
decoding: greedy / temperature 0
invalid-output policy: unchanged
paired bootstrap: 20,000 samples, seed 0
```

## Results

| Profile | C-Eval | GSM8K MC | Invalid outputs |
|---|---:|---:|---:|
| Frozen all-NPU | 27/128 (21.09375%) | 31/128 (24.21875%) | 0 |
| Aligned P1 | 27/128 (21.09375%) | 31/128 (24.21875%) | 0 |
| Aligned P2 | 27/128 (21.09375%) | 31/128 (24.21875%) | 0 |
| Aligned P3 | 27/128 (21.09375%) | 31/128 (24.21875%) | 0 |

For both benchmarks, every Hybrid profile has the same per-example
correctness vector as all-NPU.  Each comparison therefore has:

```text
NPU-correct / Hybrid-wrong = 0
NPU-wrong / Hybrid-correct = 0
absolute score delta = 0
paired bootstrap 95% CI = [0, 0]
```

Aligned P2 and P3 are also pairwise identical to their preceding aligned
profiles.  There was no parser failure, NaN, invalid-output increase, or
catastrophic subset collapse.

## Interpretation

This establishes `NO_STATISTICALLY_MEANINGFUL_REGRESSION` under the frozen
256-example contract.  It is not a general benchmark claim about
DeepSeek-V2-Lite and does not substitute for the system, deterministic, or
pairwise numerical gates.

```text
R5_ALIGNED_QUALITY = QUALITY_VERIFIED
```
