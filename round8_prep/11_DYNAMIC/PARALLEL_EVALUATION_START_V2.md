# Parallel Dynamic evaluation v2 — healthy launch checkpoint

Observation: 2026-09-19 14:49 UTC, host110.129.0.22 only.

Candidate commit before requests:
`9a183f309e1db794859646686431d9a1fbeb8819`.
Candidate SHA256:
`69892f31dcbb450105a4adf153622c7bf6f5823292bfb8038ac9ae60160e55b9`.
Threshold256/chunk512; DEV1024 skipped by explicit user instruction. No model,
mask, GGUF, tokenizer, quality sample or C64 changes.

| Lane | Placement | Controller | Observed progress |
|---|---|---|---|
| heldout | dynamic-a, NPU4, CPU0–63, NUMA0/1 |1462515|25/128|
| CMMLU | dynamic-b, NPU6, CPU64–127, NUMA2/3 |1463082|1/256, second request active|

Each completed response returned HTTP200. CMMLU first request recorded40
layer updates, proving real Dynamic execution. Initial heldout short requests
do not trigger updates; later medium/long coverage remains to be checked.
No conclusion about aggregate quality is drawn from the first answer.

Twelve resource samples per lane: cgroup swap0; OOM/oom_kill events0;
no traceback or `Check input storage_offset` warning. Container usage131.2GiB
and224GiB, each below384GiB. HostMemAvailable630GiB. Host historical swap remains
nonzero; do not claim host swap usage zero. Cgroup memory is not process RSS.

Raw paths under `/home/admin/kt-round8-qwen35`:
`qwen-dynamic-heldout-v2/` and `qwen-dynamic-cmmlu-v2/` contain entry, admission,
resolved command, raw responses, progress, resource samples and server logs.
They are live artifacts: final hashes must be taken after completion, not now.

Fresh formal Static/Dynamic performance remains gated, NOT_RUNNING. After both
complete and independent audit passes, release short64/32 and medium256/64,
three ABBA blocks on prep NPU2/CPU128–191. Never mix these parallel evaluation
latencies into formal performance. No long requests or diagnostic forward hooks
in timed tests. Heartbeat round7 stays ACTIVE every15minutes.

Local unit regression:26 tests PASS; git diff whitespace check PASS.
No push, main merge, tag, .16 deployment, or unrelated workload modification.
