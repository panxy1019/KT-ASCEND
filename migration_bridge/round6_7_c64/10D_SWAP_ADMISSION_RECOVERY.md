# Admission recovery V2

At 2026-09-17 09:16 UTC, host SwapTotal-SwapFree was consistently256KiB across
five samples. Available memory was ~985GiB, container memory.swap.current=0,
oom=oom_kill=0. The old gate rejected any host-wide swap residency, including
unattributed pages outside this workload. No OOM or local expert failure was found.
Exact historical ownership of these256KiB is unknown, not claimed as a counter bug.

Explicit revised scope: this container must have zero swap, host reserve remains
max(64GiB,20%), and if any global swap exists pswpin/pswpout must remain unchanged
across a two-second admission interval. Active host swapping still blocks. This
is a resource admission policy change, not evidence that the old gate passed.
No host swapoff, sysctl, cache purge or business process change was performed.

Completed R8 canary and eight local operator runs are retained. Only short timing
resumes into short-timing-retry1. Original failure directory/logs remain.
Driver tools-v16/resume_quality_after_admission.py automatically continues frozen
R16 and R8 CMMLU after successful short timing; final comparison remains pending.
State: /workspace/c64-dev/admission-recovery-progress.json.

Tests: idle global swap accepted only with own swap0; own swap rejects; active
host swapping rejects; insufficient memory reserve rejects.
