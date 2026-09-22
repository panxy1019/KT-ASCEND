# 1408 two-way split: not qualified on the current deployed path

Evidence state: CODE_INSPECTED / ANALYTIC_DERIVED. No C64 expert/model result is claimed.

The runbook explicitly requires stopping C64 multi-TP if `forward_many` still
uses a 256 tile for the 704 partition. That condition is present. Neither the
Python nor native TP1 guard was removed. Quantized paths were not changed.

## Actual runtime, not merely branch HEAD

Deployed `moe.hpp` SHA256:
`7aee7e59fa96da16d3096b0dbfb2de45b88257a1bca6e895f2b2fcc4071956ba`.
Local header differs only by the existing `KT_R66_STAGE_CAPTURE` test-only
instrumentation. The arithmetic under examination is identical.

`moe-tp.hpp`: `eddad2771968b73f1cb8a75ad0af5041b5ae61f125ffaa4410d8d76069a19ca2`.
`python/utils/llamafile.py`: `e480074dc2c724084c726022d28601cb9e14aec55e18b487d29958706b40a7d8`.
Unchanged C32 native extension:
`0f3aac9009931b18853c9b8fd6f6bba8b02d6ea396e79b2029bf9ce092049895`.

## Shape and arithmetic findings

| Boundary | Current behavior | C64 implication |
|---|---|---|
| Python TP validation | Non-256 intermediate requires one pool | 1408 TP2 rejected |
| Native TP constructor | Same non-256 single-TP protection | 1408 TP2 rejected independently |
| F32 weight blocking | Unblocked representation | Does not itself validate compute tiling |
| `forward_one` | `m_block=32` | 704 has 22 complete tiles |
| `forward_many` gate/up | `m_block=QK_K=256`, integer tile count | 704 would cover 512 and omit 192 channels |
| Existing 1408 prefill | `group_min_len=chunk+1` forces row-wise `forward_one` | Existing C32 is not evidence that TP2/prefill is supported |
| Down projection | Each TP part computes its partial down output | Requires same-operand split oracle |
| BF16-visible down boundary | Each TP part rounds down before routing accumulation | Cannot assume equivalence to full-expert down rounding |
| TP merge | Adds already routed partials | Too late to reconstruct individual expert down rounding |

`tools/audit_split_contract.py` supplies a deterministic arithmetic counterexample:
rounding each partial then summing differs from rounding the complete down output.
It is **not** evidence of a measured model failure or of local rel-L2 exceeding
0.01. It establishes why simply removing guards is insufficient.

## Required implementation work before any C64 promotion

1. Scope F32-only tail-safe batching/splitting separately from quantized alignment.
2. Specify and implement full-expert down boundary across TP parts, or explicitly
   qualify a new numerical profile; do not silently inherit C32 semantics.
3. Prove gate/up row slices and down-column slices reconstruct original weights.
4. Run C32/C64/canonical/NPU same-operand decode and prefill tests; 10 exact repeats,
   finite and rel-L2 <=0.01, plus negative shapes and captured partial merge.
5. Only then run R16 compact/restart/performance, followed by R8 and RAM admission.

Current decision: `C64_NOT_SUPPORTED_FOR_CURRENT_F32_PATH`.
This is a capability stop, not a model-quality failure. Existing C32 work is not
interrupted. No C64 CMMLU, B16 pilot, or performance benefit is claimed.
