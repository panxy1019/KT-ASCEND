# Static evidence audit before Dynamic qualification

## Q32 stability on .22

Campaign A: `raw/qwen-q32-stability-22-v2/campaign-A`.
Campaign B: `raw/qwen-q32-stability-22-v3-resume/campaign-B`.
Each contains 18 requests and 1152 generated tokens. Six prompt trajectories
repeat exactly within each campaign; the two independently loaded processes'
reference token arrays are exact. Total: 2304 tokens.

The original v2 B admission failure is preserved: bounded host swap readback
exceeded the existing guard. The successful B resume did not relax that guard.

| Campaign | Samples | Cgroup GiB start/peak/end | Last-quarter range GiB | NPU2 HBM MiB start/peak/end |
|---|---:|---|---|---|
| A | 67 | 131.6 / 132.4 / 132.4 | 132.4–132.4 | 18223 / 18228 / 18227 |
| B | 66 | 132.4 / 133.2 / 133.2 | 133.2–133.2 | 18223 / 18228 / 18227 |

No recorded Docker OOM event. Both campaigns reach a measured plateau rather
than continuous growth over their final quarter. This supports bounded
lifecycle stability for this workload, not a proof of zero memory leak.
Cgroup usage is not process RSS; separate RssAnon/RssFile were not sampled by
this historical collector and must not be invented. Dynamic collects additional
cgroup swap/OOM counters, independently of these Static results.

## Static v6 timing evidence

12 cells and 24 requests are present. All cells use one exact protocol SHA:
`fed966ba94d5a37037c03b34c3e90661866765e0f4fa9ebe7ed46ef9b4c0b01f`.
Every cell has eight resource snapshots. No other listed container exceeds
10% CPU in those snapshots. This does not prove absence of host processes or
host file-transfer bandwidth contention; no full host-process census was saved.

The earlier .16 migration/transfer overlaps part of this development period.
Until transfer timestamps are reconciled with each measured window, do not
promote all v6 cells to strictly isolated performance or use them as the sole
Dynamic comparator. Preserve all cells without cherry-picking.

The upcoming Dynamic ABBA therefore reruns Static and Dynamic together under
the same corrected runtime, NPU2/CPU128–191, short64/32 and medium256/64 protocol,
with diagnostic hooks absent. Parallel held-out/CMMLU lanes must be stopped
after their own completion before this comparison begins.
