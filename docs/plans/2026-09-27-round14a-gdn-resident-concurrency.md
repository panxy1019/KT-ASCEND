# Round14a execution plan

Round14a starts from the exact Round14 S0 runtime snapshot and has three
independent device lines.  It does not reopen the Round14 public-proxy quality
review, Native TP8, OffloaderV1, H512/H768 promotion, NZ update work, Dynamic,
or deferred execution.

The binding order is:

1. P0: freeze the Round14 S0 identity, source gates, resource contract, budget,
   and entry points.
2. A: identify the owner and lifetime of the repeatable 171,424,768-byte GDN
   allocation increase, make the smallest lifecycle fix, run at least twenty
   same-shape/chunk/decode/cancel-reuse cycles without `empty_cache`, then apply
   the numerical, resource, paired-component, smoke, and request ABBA gates.
3. B: create two fresh S0 five-bucket attribution blocks; calculate safe extra
   resident capacity with A and concurrency-4 reserves; screen at most
   `FPLUS(k_safe)` and an equal nominal-capacity `U_PLUS`; send at most one to
   four-process request ABBA.
4. C: audit and repair per-request state ownership and the current
   `batch_size == 1` gate; prove engine in-flight concurrency and isolation at
   c2 before c4; run the frozen four-process tier order if safe.
5. Combine only independently promoted and resource-compatible changes, run a
   final mirror only when needed, freeze evidence, retire project file cache,
   and push the completed Round14a branch.

## Frozen gates

- GDN keeps the Round14 implementation-2 semantics: NPU convolution and
  recurrence plus local sigmoid gate, RMSNorm, and output projection.  The
  rejected swish-parent implementation remains rejected.
- GDN request promotion requires both component blocks to improve the complete
  parent by at least 15%, both request blocks to reduce W1 TTFT by at least 15%
  and W1 E2E by at least 10%, with W2/W4 quality and performance protection.
- Incremental residency requires both request blocks to reduce W2 TPOT and
  increase complete serial output throughput by at least 5%, without numerical,
  state, W1, or W4 regression.
- Concurrency must be visible as engine batch/running overlap.  c2 aggregate
  throughput must be at least 1.20x c1; c4 must be at least 1.10x c2 and 1.30x
  c1, subject to per-request latency, quality, cancellation, and resource gates.
- Formal inputs remain at most 128 tokens and outputs at most 128 tokens.

## Evidence boundary

The user referenced `40_A线_GDN分配定因与性能闭环.md`,
`41_B线_五桶归因与增量前层常驻.md`,
`42_C线_并发2到4的隔离与吞吐.md`, and
`43_Round14a_执行合同与预算.md`.  They were not present in the local tree or
the fetched GitHub refs at P0.  The complete Round14a user message is therefore
the binding execution contract.  Missing attachments are recorded as evidence,
not treated as a reason to stop.

## Hard limits

The round has independent maxima of 30 model groups, 752 generation attempts,
24 abort controls, 32 device component processes, 72 wall-clock hours, and 64
NPU device-hours.  The provisional stage split in
`artifacts/round14a/ROUND14A_EXECUTION_BUDGET.json` may be rebalanced only while
preserving every not-yet-started paired block and the round-wide hard limits.
There are no blind retries.  The same root cause receives at most two modified
validation revisions.

Every device process receives a fresh 2x60-second admission.  Physical 12,
card 6/chip 0, C128, and NUMA 2/3/4/7 remain fixed.  Unknown device processes,
identity drift, or resource-gate failure is fail-closed.  Cleanup and full
evidence hashing precede targeted project cache retirement.
