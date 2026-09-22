# Formal performance resource recovery

2026-09-18. Attempt `cache-static-dynamic-abba-v1` is retained, not promoted.
Six measurements completed before the owned runner was interrupted. A real
host swap-in counter increase occurred within short/block0/turn3. Other
completed system, held-out and quality gates are unaffected.

The old observer compared cumulative `/proc/vmstat` pswpin/pswpout against its
startup snapshot forever. After any swap activity this prevented recovery even
when later intervals were quiet. The repair compares adjacent snapshots and
still resets the full60-second quiet period on any activity. Historical samples
and eligibility are not rewritten. `/proc/vmstat` field reference:
https://man7.org/linux/man-pages/man5/proc_vmstat.5.html

The new campaign waits for a fresh EXCLUSIVE observation after each warmup,
outside the measured request. This is admission only: the full start/end window
audit remains mandatory and any mid-measurement interference still rejects
formal eligibility. No business workload or host swap configuration is changed.

Validation:120 local pure tests passed, including activity recovery, fresh/stale
admission and rejection of mid-request contention. A3 execution and deployment
are separate evidence. Threshold1024, corpus, masks, native/runtime,36-cell
3ABBA matrix, token budgets, scoring and throughput definition are unchanged.
