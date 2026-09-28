# Round14a current execution

Status: `ROUND14A_COMPLETE_GDN_PROMOTED_B_C_NO_PROMOTION`

Round14a device work is complete. Physical 12/card6/chip0 is empty, every owned container was removed after guard exit ACK, registered project file cache was retired only after archive/SHA, and historical timers remain paused.

## Final decisions

- A/GDN: promoted. The 171,424,768-byte observation was an old eager-reference component-probe autograd false positive, not production per-layer workspace. Production-equivalent lifecycle, numerical/component gates, two smoke groups and K0/KC/KC/K0 request ABBA passed. Both blocks reduced W1 TTFT by more than 53% and W1 E2E by more than 36%.
- B/incremental residency: no promotion. `k_safe=1`; FPLUS1 adds 4.39453125 GiB nominal BF16 residency but improved W2 serial throughput by only 2.55% and 3.99%, below the frozen 5% gate. Default remains Uniform Q32.
- C/true concurrency: no unified default promotion. C2/C4 quality, request-state isolation, cancel-peer safety, reuse, and true engine batches passed. W2 C4 reached 5.068–5.284 tok/s, but W1 and order-stability gates failed across all four formal blocks. Default remains concurrency1.

## Final identity

Promoted GDN + Uniform Q32 + C128/MXFP4 + BF16/NZ29 + chunk32 + PLE ASYNC + QSA Vectorized Prefill/DIRECT_BF16_ROW Decode + H129 + TP1 + concurrency1. Dynamic/deferred/R2/R3/transpose-view/shadow320 remain off. Qualified request scope is input <=128 and output <=128 tokens; broad product quality is not claimed.

## Budget

23/30 model groups, 535/752 generation attempts, 4/24 abort controls, 10/32 component processes, and 3.5959/64 NPU device-hours.

## Delivery state

Final report, benchmark, results, budget, resource retirement, final verification and SHA manifests are frozen under `artifacts/round14a/`. The device and evidence phases are complete; delivery uses an exact Round14a whitelist so the 1.4 GiB duplicated runtime snapshots and unrelated historical dirty worktree are not committed. No further device experiment is required.
