# Bounded range throughput screen — 2026-09-20 UTC

User requested reduced segmentation overhead and evidence for two connections.
Only owned download service was paused/restarted; unrelated traffic was untouched.

## Implementation and protocol

The old guard slept five seconds even if curl completed earlier. It now waits
on child completion with a five-second timeout, retaining resource checks while
returning immediately on completion. Regression reproduced 5.002s before fix;
all 14 tooling tests now pass, including two-worker overlap/ordered publication,
range rejection, no-clobber SHA publication and tiny-tail handling.

Commit 63ceafadb froze the benchmark before execution: single/dual/dual/single,
128MiB each, same first four 32MiB ranges of frozen shard48. Timing includes
fetch, range validation, ordered append and fsync. Source revision remains
de4b8e4d43b917e7706784d8bb445c9af86a3540. Selection required both paired gains
positive and geometric gain >=1.10, with no recovered transport errors.

| Cell | Connections | Seconds | Decimal MB/s |
|---|---:|---:|---:|
| A1 | 1 | 25.4573 | 5.2723 |
| B1 | 2 | 19.9234 | 6.7367 |
| B2 | 2 | 23.2466 | 5.7737 |
| A2 | 1 | 25.5919 | 5.2445 |

Paired gains: 1.277762 and 1.100889; geometric gain 1.186033.
All assembled prefix hashes equal
8b2084def7b4560b71132278350e38032c0c5bdd4e03b6bf73d685c018f30f5a.
Zero transport retries. Selected two workers. This is bounded screening under
shared network contention, not a guarantee of sustained speed or NIC capacity.
It isolates connection count after the polling fix, not the independent total
benefit of that fix. Prefix identity does not replace full model SHA audit.

## Tail failure and minimal repair

On resumption shard50 had only176bytes remaining. Curl exited63 three times
at resolver HTTP302, before final206. Per-range max-filesize176 was smaller
than the redirect response. Preserve headers/failure JSON in remote stage.
[curl documentation](https://curl.se/docs/manpage.html#--max-filesize)
documents the size limit and exit63. Commit4bd8f3f90 retains a bounded32MiB
transport ceiling even on tiny tails; final Content-Range, exact body length,
and complete official SHA still gate acceptance. Regression failed before fix
and passed afterwards. Real tail then succeeded and entered shard verification.
Full shard50 SHA then PASS:6014a2b71b74a5e2703bf5a1182251a71d7f0720a0244988874deb430b312d50.
Production shard51 committed402653184bytes in57.8831763s (6.956MB/s), without
restarting the live downloader. Published50shards totaling148352668760bytes.
Whole-model SHA audit remains pending. Cross-shard parallelism not evaluated.

## Artifacts and runtime

Local result: evidence/range-throughput-abba-v1.json.
Raw remote: /home/admin/kt-round9-qwen38/range-throughput-abba-v1/
(protocol, headers, prefixes, results and decision).
Active service: kt-round9-range-recovery-v2, KT_RANGE_WORKERS=2.
Same proxy and CPUQuota400%, MemoryMax8G, swap0, Nice19, idle I/O class.
Production retains300MB/s disk caps; benchmark had no disk bandwidth cap,
well above observed network-limited rates in either case. No qdisc/route changes.
