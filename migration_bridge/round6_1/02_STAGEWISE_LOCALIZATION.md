# Round 6.1 stagewise localization

Evidence state: `A3_VERIFIED`.

This work package localizes the immutable Round 6 V1 held-out failure
`hd_en_02:43` using the same 54-token matched history. It does not change the
placement, weights, model, tokenizer, Deferred budget, or rejected numerical
contract.

## Frozen executions

| Execution | Meaning | Final top-1 | Margin `21 - 16520` |
| --- | --- | ---: | ---: |
| A | production `defer=0` | 21 | 10.75 |
| B | production `defer=1` | 21 | 11.5 |
| C | explicit diagnostic replay | 21 | 11.5 |

The stage dumper used per-stage call indexes because one serving request can
perform warm-up/control calls in addition to the 54-row matched-history call.
The target is `call0003`, selected by its `[54, 2048]` hidden-state shape and
the frozen request identity, rather than by assuming that every request maps to
one internal forward.

## First divergence

At layer 7, the following target-call tensors are exact between A and B:

```text
decoder input hidden
attention output
MLP input
router logits
top-k IDs and ordering
routing weights
NPU-owned routed partial
```

The first non-exact tensor is the layer-7 CPU partial:

| Tensor | dtype / shape | max abs | relative L2 | Interpretation |
| --- | --- | ---: | ---: | --- |
| L7 CPU partial | FP32 `[54,2048]` | 0.0991039 | 0.136574 | B intentionally omits the selected deferred contribution at L7 |
| L7 routed merge | BF16 `[54,2048]` | 0.1015625 | 0.0207769 | direct visible consequence |
| L7 MoE visible output | BF16 `[54,2048]` | 0.1015625 | 0.0158147 | value passed to the next layer |

This difference is predicted by the Deferred schedule. It is not a different
router decision, expert weight, NPU expert result, or CPU GEMM result.

## Layer 8 and propagation

The delayed L7 contribution is consumed once as the L8 incremental input. By
then A and B already have different L8 inputs, so differences in L8 attention,
router logits, four top-k entries, NPU partial, and CPU partial are downstream
effects rather than the primary divergence.

Selected amplification checkpoints are:

| Checkpoint | max abs | relative L2 |
| --- | ---: | ---: |
| L8 decoder input | 0.1015625 | 0.0158147 |
| L8 attention output | 0.0166016 | 0.0151119 |
| L8 router logits | 0.0964811 | 0.00395372 |
| L8 routed merge | 0.105469 | 0.0335881 |
| L12 visible MoE output | 0.1328125 | 0.0221635 |
| L17 visible MoE output | 0.162109 | 0.0400952 |
| L22 visible MoE output | 0.375 | 0.0348183 |
| L26 visible MoE output / final norm input | 1.25 | 0.0114216 |
| final norm output / LM-head input | 0.2578125 | 0.0178363 |
| final full logits | 0.8125 | recorded in evidence |

All captured tensors are finite.

## Classification

```text
PRIMARY_FIRST_DIVERGENCE = DEFERRED_VISIBILITY_DIFFERENCE
FIRST_LAYER              = 7
FIRST_OPERATION          = CPU routed partial visibility
FIRST_TENSOR             = ktep_layer07_cpu_partial_call0003.pt
EXPECTED_REASON          = selected L7 contribution is intentionally exposed at L8
ACTUAL_REASON            = matches expected Deferred schedule
SECONDARY_EFFECT         = DOWNSTREAM_NONLINEAR_AMPLIFICATION
```

The 0.75 final pairwise distortion is therefore not used as a root-cause
shortcut. The attribution proceeds forward from L7 CPU visibility through L8
incremental consumption and later nonlinear layers to the LM head.

## Evidence

- Normalized comparison: `evidence/stagewise-hd-en-02-43/result.json`
- Frozen requests: `evidence/stagewise-hd-en-02-43/*_request.json`
- Raw tensor index: `evidence/stagewise-hd-en-02-43/raw_sha256sums.txt`
- Raw A3 roots: `/workspace/results/round6_1-stage-v2-{nodefer,deferred,replay}`

```text
ROUND6_1_STAGEWISE_LOCALIZATION = PASS
```
