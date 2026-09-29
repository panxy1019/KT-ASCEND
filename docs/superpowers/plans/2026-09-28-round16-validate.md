# Round16 device and request execution plan

Authority: the user's Round16 concise target and detailed plan 66. This plan
continues the existing `round16-contract-v2` deliverables. It does not redo DEV
or Round15 and does not reset the existing Round16 budget.

## Checkpoints

1. **P0 terminal and compatibility:** verify Round15 terminal evidence, S15
   identity, self-cleanup, released lease, native donor identity, source gates,
   budget and safe fresh admission. Evidence: P0 receipts and exact code map.
2. **A1 current critical path:** run two fresh S0 profile blocks, separately
   account CPU Expert, NPU work, Host dispatch, synchronization and UNKNOWN,
   and freeze the dependency DAG and A2/A3 decision.
3. **A2/A3 bounded candidate:** implement at most two component candidates and
   select at most one request candidate. All new behavior is default OFF and
   follows RED→GREEN tests, reference, 20-round lifecycle and cancel/reuse.
4. **A model gate:** only a component candidate with both blocks at least 10%
   faster gets one smoke group and then `S0/A16/A16/S0`.
5. **B real integration:** bind the actual frozen vLLM-Ascend worker/model API,
   intercept before expert allocation, load only Q32 on NPU, run real Hybrid
   MoE, decoder layer, TP1 eager model smoke, then a legal graph boundary if
   useful.
6. **B request comparison:** after smoke, compare `R/B16/B16/R`, where R is
   promoted A16 or S0. Record system differences rather than calling it a pure
   runtime ablation.
7. **Final freeze:** answer current 3.x tok/s limitation, A2/A3 disposition,
   A request economics, B integration level and native fusion/graph coverage,
   paired A/B behavior, resource/quality cost and default-carrier disposition.

## Fixed gates

- W1 128/16, W2 128/128, W4 64/32; same prompt IDs, template, sampling, output
  contract, bank, Q32 map and resource admission.
- Decode promotion: both W2 paired blocks improve TPOT at least 10% and serial
  output throughput at least 10%. Prefill gate may replace it only when frozen
  before implementation from A1 evidence.
- Any non-primary protection metric regressing more than 5% in a block prevents
  direct default promotion.
- Numerical, state, cancellation/reuse, allocation and cleanup gates precede
  performance.
- Maximum 14 model groups, 128 generations, 6 aborts, 16 device components and
  32 NPU device-hours; no borrowing or reset.
- Device/model/performance jobs are mutually exclusive. External tasks are
  read-only. No push, merge, tag, reset, clean, timer or global cache drop.

## Current progress

- [x] Round15 exact terminal and all 31 final hashes verified.
- [x] Round15 self-cleanup and lease release normalized from final and current
      read-only evidence.
- [x] Barrier exact terminal spelling added through a failing-then-passing test.
- [x] Admission cycle 1 failed closed before container creation because every
      NPU has an external deployment claim or active external engine.
- [ ] Resolve S15-to-A/B source, ABI, provider and state compatibility.
- [ ] Run A1 two-block profile after admission passes.
- [ ] Decide and, if justified, validate A2 or A3.
- [ ] Progress B through real MoE, decoder layer and model smoke.
- [ ] Run required fresh paired request comparisons.
- [ ] Freeze final Round16 status and Chinese report.
