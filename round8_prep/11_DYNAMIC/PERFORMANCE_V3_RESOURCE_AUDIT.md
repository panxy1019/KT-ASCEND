# Performance v3 final collection/resource audit

Audit2026-09-19 21:20UTC, .22 only. Twelve fresh-process ABBAcells and24requests
collected; controller exited, all owned Round8 containers stopped,OOMKilled=false.
Frozen protocol identical across all12cells:short64/32,medium256/64,greedy,
ignore_eos,hook-free,graph/deferredOFF,C64/Q32 unchanged. Full ordered cell IDs,
protocol SHA,input token IDs,payloads,HTTP200,output counts and recorded timing
agreement independently verified. No watchdog/Traceback/NZ offset warning in
the completed cell logs. Historical v2 remainsFAILED and excluded.

## Descriptive observations, NOT strict isolated performance qualification

| Profile | Short median E2E seconds | Medium median E2E seconds |
|---|---:|---:|
| Static |24.078804910|47.208510328|
| Dynamic |24.335451172|472.004721214|

Six samples/profile/workload. Dynamic medium includes the actual cold full-layer
weight preparation/update, not only steady decode. No separate TTFT/decodeTPS
was measured; do not infer either from E2E. No confidence interval or universal
speed ratio is claimed. Observations do not justify enabling Dynamic by default.
Retain frozen Q32-C64 Static; DynamicOFF,DeferredOFF.

## Resource audit limitation

284resource samples. Other Docker containers sampled0%CPU, NPU process samples
contained only device2; Round9 download was explicitly paused for the campaign.
However, /proc/vmstat deltas within sampled cell windows total37460swap-in pages
and54017swap-out pages (4096bytes/page). Seven cells have nonzero swap-out:
all6Dynamic cells and Static0-0. Dynamic0-1 host memory-pressure total also
increased by approximately11.5seconds. Global MemAvailable remained ample;
this does not exclude NUMA-local reclaim or unrelated host activity.

Current Docker config preserves CPU128–191,mems4,5,384GiB memory and equal
MemorySwap (no owned container swap allowance). This alone cannot retrospectively
attribute host swap activity. Per-cgroup memory pressure/NUMA samples were not
collected by this performance monitor, and ps %CPU is lifetime-average, not
interval utilization. Do not label a paused HF process active from that field.
Maximum resource sample gap13.6193s; several final request tails extend up to
about9.1s beyond the last sample. No complete end-boundary resource sample.

Thus COLLECTION=COMPLETE,PROTOCOL_AUDIT=PASS,
STRICT_ISOLATED_PERFORMANCE=NOT_QUALIFIED_RESOURCE_AUDIT.
Do not rewrite this as PASS or silently relax zero-swap/resource criteria.
No automatic additional full12cell search for a favorable window is authorized
by this audit. Diagnose resource provenance and collector bounds before any
new timing identity; do not alter host swap settings or unrelated tasks.
Round8 final report/overall closure remains pending explicit gate accounting.

## Raw evidence

On persistent .22 root /home/admin/kt-round8-qwen35 and local round8_prep/raw/:
qwen-dynamic-perf-v3-complete.tar.gz
SHA256:8ac9bb1b59d641a8e867414ce78e20466e9e77d4d5475fba3cf83e0781f6946e.
Archive includes full v3collection and frozen retry configs/ack. Original
state.json remains COLLECTED_PENDING_RESOURCE_AUDIT; this separate audit records
the limitation without rewriting raw evidence.

Round9 pause marker removed only after controllerinactive and all3owned lanes
confirmedstopped. Resumed download then failed with TLS ReadError
DECRYPTION_FAILED_OR_BAD_RECORD_MAC; originaljournal preserved. Restarted same
owned service with samepinnedrevision/cache/resourcecaps, no force-download,
no certificate-validation bypass. Full model identity remains unverified.
