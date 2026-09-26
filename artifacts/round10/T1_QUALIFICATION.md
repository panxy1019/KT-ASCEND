# T1 Hybrid narrow qualification

Status: `PASS_T1_H0_ROUND10_NARROW`.

The qualified scope is serial single-request Hybrid execution with actual post-template input length at most 128 tokens and a natural-generation limit of 64 tokens. Five deterministic canaries passed across preserved v93/v95 evidence. Live 63/64/65/127/128-token requests all returned `[3897, 248046]` (`ACK`, EOS) through real Hybrid CPU and NPU provider events. The two decode cases produced 32 tokens to the length boundary and 49 tokens to EOS with finite per-token log probabilities.

Two independent 128-token mid-prefill cancellations triggered at chunk start 32. Both reached abort, published the submit fence, retired the request, committed cleanup, showed no late commit, returned the lease, and were followed by an exact same-process `391` reuse response. The supervisor obtained guard exit acknowledgement and removed the owned container. Physical12 was empty afterward.

Resource maxima were 342.10 GiB owned host cgroup, 28778 MiB board HBM, 25762 MiB owned process HBM, and 24720.62 MiB allocator peak, all within the frozen limits.

The 129/130 policy predicate is component-tested as Layerwise, but no new live `L-PROBE` was spent. Therefore request totals 129 and above remain outside this Round10 Hybrid qualification. T1 is a policy/quality result; the H64 median of 5.188200 seconds is recorded for continuity and is not a new OFF/ON performance claim.
