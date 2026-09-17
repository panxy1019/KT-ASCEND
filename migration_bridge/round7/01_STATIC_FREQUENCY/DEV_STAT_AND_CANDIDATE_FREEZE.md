# DEV complete and Static Frequency candidate freeze

Date: 2026-09-17 UTC. Evidence: A3_VERIFIED. This freezes a mask candidate,
not system, quality or performance acceptance.

## Integrity gate

| Item | Observed |
|---|---:|
| DEV requests | 256/256 |
| Frozen prompt token count | 245,872 |
| Generated token budget | 16/request |
| Raw model forwards | 4,352 |
| Attributed forwards | 4,352 |
| Effective forwards | 4,096 |
| Proven unused final-token forwards | 256 |
| Unknown forwards | 0 |
| Effective assignments per routed layer | 1,498,272 |
| Total effective routed assignments | 38,955,072 |
| Clean logical_count shape | [4096,27,64] |
| Dense layer0 assignments | 0 |
| Finite, nonnegative counts | PASS |
| Observer ON/OFF pilot parity | PASS |
| Owned server lifecycle/reaping | COMPLETE |

Completeness identity: `(245872 + 256*15)*6 = 1498272` per MoE layer,
identical across all 26 routed layers. Every raw forward matched exact prompt
input IDs/positions or the returned baseline generation history. Extra passes
are preserved, not silently discarded or counted as evaluated positions.

## Official Frequency candidate

The actual deployed `kt_kernel.generate_gpu_experts_masks` selects global top208
from DEV activation counts. The integrated loader computes the same routed
budget using 26 actual MoE layers. Dense layer0 does not consume the budget.
No HOLDOUT, CMMLU labels, historical failure trajectory or numerical-discrepancy
scores participated in selection.

Per-layer resident count, layers1 through26:

```
10,7,11,8,9,8,8,8,9,10,12,9,10,6,5,9,5,5,9,8,6,4,7,8,8,9
```

Total208; range4–12. Zero-resident layers: none in this derived candidate.
Zero-slot compatibility therefore remains an unexercised model-level edge
case, not a fabricated PASS. Per-layer counts must not be changed to force8.
Full logical IDs and ownership partitions are frozen in the JSON below.

## Artifact freeze

- DEV corpus SHA: `b79c132b095a0efd35c822f1b1124e57b666b1c5ffdf65f4933e7bcad02c694f`.
- Clean activation statistics and identical frequency.pt bytes SHA:
  `e54c0c137738809a67ee04820104e46808f207ea1afd2470c8133458fd816690`.
- Frequency JSON SHA:
  `da95010893e9e90b109b23269b052cd5b465d844985579ab76c93425d5f2f05e`.
- Complete byte-frozen DEV recovery archive SHA:
  `297d39704f81a607e8c389ce9ca1702d81b94cf8b7551fa3e23aa367ca04ed37`.

Archive: `../raw/dev-stat-evidence-v2.tar.gz` (about12MiB compressed). It contains
the full runtime-v2 snapshot: original recorder dump, raw and clean .pt files,
exact frequency.pt bytes, requests/logits metadata, all forward input-ID/position
metadata, route traces, source provenance, lifecycle, pilot and candidate masks.
The .pt files are compressed losslessly, not reconstructed from summaries.
Embedded frequency.pt was streamed out of the archive and SHA-checked locally.

Readable copies:

- `../raw/dev-stat-v2/runtime-v2/dev-stat-summary.json`
- `../raw/dev-stat-v2/runtime-v2/profiles/frequency.json`
- `../raw/dev-stat-v2/runtime-v2/frequency-candidate-freeze.json`
- `../raw/dev-stat-v2/runtime-v2/uniform-dev-v1/forward-attribution-epoch-2.json`
- `../raw/dev-stat-v2/runtime-v2/uniform-dev-v1/lifecycle.json`

Only logical_count is filtered for frequency fitting. Other recorder fields
remain original evidence and are not asserted to be aligned clean per-pass data.

## Next gate

After this candidate is in a local commit, send an explicit commit/hash ack to
the serial driver. Then run representative operators, Uniform/Frequency compact
and restart/parity, independent HOLDOUT routing, fixed CMMLU256 and observer-OFF
isolated ABBA. Frequency has not yet passed those gates. Dynamic and Deferred
remain OFF, historical strict numerical blockers remain unchanged.
