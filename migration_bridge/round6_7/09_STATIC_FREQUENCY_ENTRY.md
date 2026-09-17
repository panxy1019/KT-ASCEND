# Static Frequency / Dynamic entry boundary

Current status: NOT_STARTED; resident-ratio screen running.

After SYSTEM + CMMLU product PASS and measured capacity/performance review,
select one primary resident budget from R32/R16/R8. Static Frequency gets new
SCHED_DEV and independent SCHED_HOLDOUT, freezes masks before evaluation, and
compares against the same-budget uniform baseline. CMMLU quality answers do
not select hot experts. CPU-route reduction and fully-NPU token fraction are
primary scheduling signals; per-layer and whole-model token denominators must
be distinguished. Count ambiguity/cardinality or legacy numerical divergence
only as diagnostic under this product policy.

Only after this strong static baseline is qualified can Round7 compare Static
Frequency with Dynamic Frequency Update. Round6.7 does not implement migration,
nor reopen Deferred (default remains OFF). R4/R0 are not automatically run.
