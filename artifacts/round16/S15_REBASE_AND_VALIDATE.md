# S15 rebase and validation procedure

This procedure is inactive while `ROUND15_BARRIER.json.ready` is false. It is
not a background watcher and does not reserve a device.

## 1. Open the barrier once

Read the terminal Round15 receipt and pass its normalized fields to
`round16.common.round15_barrier.evaluate_round15_receipt()`. The receipt must
identify a terminal status, a SHA-256 baseline manifest, a non-empty
code/patch/dependency identity, cleanup PASS, zero owned controller/model/guard
processes, a released resource lease, and no open baseline-correctness threat.
`PAUSED`, `RUNNING`, `IN_PROGRESS`, missing fields, or a stale lease keep the
barrier closed.

## 2. Build S15 compatibility identities

- `S15` is the last trusted default from the terminal receipt.
- `A16` combines S15 with `round16.a`; only an actually promoted Round15
  provider is enabled. If S15 already implements identical compute, record
  `REUSED_ROUND15_PROVIDER` and do not claim a new A speedup.
- `B16` uses the same model revision, MXFP4 bank identity, Uniform Q32 map,
  BF16/NZ29 representation and mathematical provider contract through the
  model-scoped vLLM bridge. ABI-specific native code is rebuilt for the chosen
  image instead of reusing a mismatched `.so`.

Rerun every affected offline test after resolving the real symbols. A changed
contract gets a new version; it never silently replaces `round16-contract-v2`.

## 3. Select resources after the barrier

The user authorized another safe single NPU and NUMA set for Round16. Select
them only through fresh admission using S15's latest, possibly stricter caps.
Verify external training container ID/cgroup, all NPU processes, CPU SMT
siblings, NUMA startup and running headroom, system memory, allocator, owned
HBM and whole-card HBM. No automatic fallback to physical 12 or historical
NUMA 2/3/4/7 is allowed when the selected resource fails admission.

## 4. Validation ladders

### A

1. Real primitive with real weights and the module-specific reference.
2. One complete GDN or QSA layer.
3. Twenty same/changed-shape, cancel and reuse cycles.
4. Full-model smoke: five canaries, multistep Decode, cancel and reuse.
5. `S15 → A16 → A16 → S15` request comparison, profiler off.

Metadata creation, conversion, allocation, event wait and teardown belong to
the measured boundary. Eager semantics qualify before any segmented graph.

### B

1. Resolve the frozen vLLM API and prove Q32 allocation before any full expert
   allocation.
2. Real hybrid MoE component.
3. One decoder layer with QSA/GDN/PLE state reference.
4. Single-NPU eager full-model smoke with CPU/resident/native-path hits.
5. Only then investigate legal segmented graphs, proving fresh CPU work per
   replay.
6. `A16 → B16 → B16 → A16` system comparison, profiler off.

An integration failure remains `B_INTEGRATION_GAP`; it is not a no-gain result.

## 5. Locked budget and metrics

The shared maximum is 14 model groups, 128 generation attempts, six abort
controls, 16 device-component processes and 32 NPU device-hours. W1 is
128/16, W2 is 128/128, and W4 is 64/32. Record actual output tokens, TTFT,
TPOT, E2E, continuous-window output tokens/s, failures, CPU wait, NPU events,
graph hits/fallbacks, host memory and owned/whole-card HBM.

The Decode gate requires both W2 paired blocks to reduce TPOT by at least 10%
and increase output tokens/s by at least 10%. A Prefill gate must be selected
before results and requires both W1 blocks to reduce TTFT by at least 15% and
E2E by at least 10%. Quality, state, cancellation and resource gates precede
performance.
