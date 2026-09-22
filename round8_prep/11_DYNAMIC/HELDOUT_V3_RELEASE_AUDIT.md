# Held-out v3 release audit — 2026-09-19 18:29 UTC

The explicit retry adapter verified the immutable candidate, original config,
retry config/runner/ack hashes and preservation of FAILED heldout-v2 before
running the frozen complete-input auditor against v3 and CMMLU-v2.

Held-out:128/128 ordered unique frozen requests, exact input IDs and complete
payloads, HTTP200, matching response IDs and16output tokens each. 96/128 requests
triggered Dynamic.8320update events cover40layers with208generations each;
weights/mapping/mask/wrapper-mask generation equality, exact weight identity and
mapping assertions pass. Consumer generation consistency also passes.
No resource-sampled owned swap/OOM or server Traceback/storage_offset warning.

CMMLU re-audit:224/256,invalid0,+0.78125pp against Q64 reference222. Both gates
PASS. These are system/product screens, not throughput or universal numerical
equivalence claims. Historic watchdog failure remains unchanged.

| Held-out artifact | SHA256 |
|---|---|
| summary.json | a24e3c16c305bcaf9f546fc6dd276bc4acbf8ae705e346ef1da00bccd2ffe7f1 |
| raw.jsonl | 3dfee9361091aab08ce4f41d374ade13e2ac520c6fc756c7203a22bf370b73b3 |
| resources.jsonl | 5e65ab65ba0abc4815e68311819528878de54b69dc8cbb0734413c09f40d8535 |

Combined raw/config/retry archive on persistent .22 host:
`/home/admin/kt-round8-qwen35/qwen-dynamic-qualified-v2-v3.tar.gz`.
SHA256:b4521fb701e52bc251d582f133817d3b373cd4fe08c1b2c48ea6778d1a149c0e.
Archive created before release publication; original collection state files
remain intact. Separate release is `qualification-release-v3.json` here, copied
to candidate directory as `qualification-release.json` after review.

## Performance dispatch

Round9 PAUSE_DOWNLOAD marker set, guard confirmed PAUSED and child process
group stopped (T). No NPU processes before launch; unrelated containers had
0%CPU and were not changed. Host quiet vmstat samples retained in session;
formal per-cell admission remains mandatory, including swap-delta policy.

Dispatched owned systemd service `kt-round8-dynamic-perf-v2` using
`qwen_dynamic_retry_release.py --performance`. Frozen protocol:three ABBA
blocks,12fresh-process cells,short64/32 andmedium256/64,hook-free,NPU2,
CPU128–191/NUMA4,5,C64unchanged. No long requests. Collection and final resource
audit remain PENDING. Keep Round9 paused for the entire campaign, including
between cells; remove marker only after campaign exits and no timed work remains.
