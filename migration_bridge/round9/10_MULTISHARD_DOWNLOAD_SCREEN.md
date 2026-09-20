# Four shards × two connections screen

2026-09-20 UTC. User requested actual eight-connection testing to save download time.
Protocol frozen in0e597f68e before execution. Same frozen revision, official
endpoint and local proxy as production. Same four files48–51, first64MiB each.
Each cell256MiB; ABBA total1GiB. Baseline downloads these files sequentially
with two ranges concurrent; candidate four files concurrent, two ranges each.
Timing includes fetching, exact Range validation, ordered fsync and prefix hash.

| Cell | Concurrent files | Connections | Seconds | MB/s |
|---|---:|---:|---:|---:|
| A1 | 1 | 2 | 38.5099 | 6.9706 |
| B1 | 4 | 8 | 31.9438 | 8.4034 |
| B2 | 4 | 8 | 34.2974 | 7.8267 |
| A2 | 1 | 2 | 36.7334 | 7.3077 |

Geometric paired gain1.13630 (+13.63%); individual gains20.56% and7.10%.
Zero retries; all four per-file prefix hashes match across all cells. No full
shard publication from benchmark.16 tooling tests pass. This is a short shared
network screen, not eight-connection long-run qualification or peak NIC speed.
The experiment changes file concurrency and total connections together; it
does not establish an independent causal benefit of cross-file scheduling.

Raw remote /home/admin/kt-round9-qwen38/multishard-abba-v1 retains protocol,
headers, bodies and decisions. Local evidence/multishard-abba-v1.json records
all timings and hashes. Same CPUQuota400%, MemoryMax8G, swap0, idle I/O class.
No unrelated workloads, routing or network priorities changed.

Eight connections passed the prespecified exploratory selection rule, but the
user requested a test; production downloader still uses two connections after
resumption. A robust four-file production controller must preserve independent
resume/progress, per-file SHA publication, and non-colliding status updates.
Do not start four copies of the current controller: its global lock forbids it.
At equal remaining bytes, the observed aggregate times suggest~12% shorter
transfer time, not4x speed. Extrapolation excludes final identity audit/retries.
