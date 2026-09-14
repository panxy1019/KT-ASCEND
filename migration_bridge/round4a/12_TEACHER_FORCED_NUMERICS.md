# Teacher-Forced Numerical Attribution

P1 is deterministic, and the first divergent `v_en_01` generation shares the
same first generated token (`185`) in all-NPU and Hybrid. Consequently the
second-token comparison uses the same exact token history.

Hybrid top candidates at the first divergent step were:

```text
token 549: logprob -2.330110549926758
token 185: logprob -2.392610549926758
top1-top2 margin: 0.0625
```

The matching all-NPU candidates on the exact same history were:

```text
token 185: logprob -2.2814865112304688
token 549: logprob -2.3439865112304688
top1-top2 margin: 0.0625
```

The candidate gap moves by exactly `0.125`: all-NPU favors token 185 by
`0.0625`, while P1 favors token 549 by `0.0625`. This proves the earliest token
divergence is a deterministic near-tie flip on an identical history, before
later autoregressive history divergence amplifies the maximum logprob delta.

The controlled layer proof and expert identity matrix exclude routing-weight
duplication, shared-expert duplication, global/local mapping corruption and
nondeterministic staging as the immediate cause. The supported candidate root
cause is amplification of the measured LLAMAFILE versus BF16 NPU expert-path
numerical difference under 1,233 high-frequency P1 CPU hits.
