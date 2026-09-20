# Download timeout repair and transport selection

2026-09-20 UTC. User authorized autonomous downloader repair and speed screening,
then additionally requested a bounded download test on110.123.0.3.

## Findings and repairs

Old32MiB ranges had a90s total deadline. Retained failed bodies contained19–23MB
before curl28; a retry downloaded the same range from its original start. Reducing
16connections to8 briefly recovered progress but did not remove later timeouts.
Memory was below its16GiB cap and OOM counters stayed0. Concurrency was not proven
to be the sole root cause of the changing network throughput.

The first targeted fix uses a600s hard cap with a1024B/s low-speed threshold over
60s. Live requests continued receiving bytes beyond90s without being discarded.
This fixes premature interruption, not external congestion. Reference semantics:
[curl speed-limit / speed-time](https://curl.se/docs/manpage.html#--speed-time).

A second concrete defect: range URLs hardcoded huggingface.co despite the approved
endpoint configuration. URLs now honor HF_ENDPOINT, restricted to official and
hf-mirror.com; revision and repository remain frozen. Existing whole-file SHA
publication and final model audit remain mandatory. A mirror8MiB prefix matched
the published official shard61 prefix exactly, including final206 range identity.

Both regression checks failed before their respective fixes; all20 tooling tests
pass afterward. No model/runtime code was changed.

## Transport observations

| Test | Bytes / outcome | Throughput |
|---|---|---:|
| Local proxy to official |4.45MB before35s timeout|0.127MB/s|
| .22 official direct |Connection timeout;0bytes|Unavailable in test|
| .22 mirror direct, identity sample |8MiB in5.796s, exact prefix|1.447MB/s|
| .22 production mirror,8connections |2.906GB committed in180.283s|16.12MB/s|
| .3 mirror single connection |16MiB in4.514s|3.717MB/s|
| .3 mirror8connections |128MiB in10.640s, all206/exact length|12.615MB/s|
| .3 official direct |Connection timeout;0bytes|Unavailable in test|

These are shared-network exploratory samples, not simultaneous controlled
benchmarks. .3 small probes establish connectivity/throughput only; they do not
constitute model identity verification. .22 production already reaches comparable
or higher throughput, so no model relocation or .3 persistent worker was started.

## Selected background service

.22 kt-round9-range-recovery-v6, PID2594900,4shards x2ranges,32MiB ranges.
Direct HTTPS mirror; proxy variables empty and NO_PROXY=* scoped to this service.
CPUQuota400%, MemoryMax16GiB, swap0, Nice19, idle I/O, read/write caps300MB/s.
Old v4/v5 stopped; existing checkpoints, failed bodies and failure JSON retained.
No host-wide networking or unrelated workloads modified. Heartbeat stays PAUSED.

Script SHA256 f1a2f7ca1b1d4499ac2dce87ccd2b127868fe249b07dd516825a6ba3f1dff5b2.
Frozen revision de4b8e4d43b917e7706784d8bb445c9af86a3540.
Model path /home/admin/kt-round9-qwen38/models/<revision>.
Stage path /home/admin/kt-round9-qwen38/range-recovery-v1.

02:26:05 saved193661071464/360000192888bytes;64/131complete files.
Newly published shards62 and63 passed official SHA256. No new failure since
mirror service start; OOM/max counters0 at02:26. Final whole-model audit PENDING.
Raw per-range headers and receipts remain in remote stage. Local machine-readable
sample and configuration record: evidence/download-slowfix-mirror-20260920.json.

Useful speed is committed bytes, excluding duplicate hardlinks and in-flight
partial bodies. The observed rate implies about2.9hours remaining transfer at
this snapshot, but shared-link variation and final verification add uncertainty.
