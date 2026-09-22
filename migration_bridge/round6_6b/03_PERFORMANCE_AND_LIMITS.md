# Observational timing, not a release performance benchmark

| Workload | All-NPU request seconds | four-R32 request seconds | Generated tokens N/H |
|---|---:|---:|---:|
|32GSM8K|349.3603|1309.2929|3762/3254|
|256CMMLU|49.0665|1917.1652|256/256|

These are summed HTTP request durations,not total job wall time,device kernel
time or a controlled ABBA performance campaign. GSM trajectories/output lengths
differ,so the ratio is not a matched-token decode-speed estimate. CMMLU uses
identical prompts and a one-token likelihood query:the observed request-time
ratio is about39.07×. This exposes substantial F32 CPU prefill cost at these
few-shot input lengths. It does not identify memory bandwidth saturation without
bandwidth/counter evidence,and does not justify changing workers during a
numerical qualification run.

Status:`PERFORMANCE_REVIEW_REQUIRED`;no speedup claim. Target shared-load effects
were not controlled as in a release benchmark. Runtime stayedTP1,batch1,NPU0,
32workers,one pool,four-R32,production overlap,Deferred/Dynamic/Graph OFF.

Read-only container observations during execution were27.34 and27.35GiB out of
192GiB,with about30–32CPU cores active. These few observations are not a proof of
zero leak or a complete RSS/HBM stability campaign. Both model lifecycles ended
normally,and the owned outer guard reportedEXIT_ZERO with complete reaping.

The quality screen failed independently of performance. Neither slow execution
nor statistical uncertainty can retroactively convert its frozen result toPASS.
