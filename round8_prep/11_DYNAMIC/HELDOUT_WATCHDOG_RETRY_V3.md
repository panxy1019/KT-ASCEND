# Held-out v2 failure and bounded v3 retry

2026-09-19: v2 failed on `holdout_032`, after32 short requests. At14:57:37
server emitted `Scheduler watchdog timeout (self.watchdog_timeout=300,
self.soft=False)`. Per-layer preparation continued through layers38/39 and
the next chunk's layer0 before SIGQUIT. Client observed connection abort.
This is a non-timed cold layerwise-prefill watchdog failure, not evidence of
a mapping assertion, transfer warning or OOM. Resource samples: cgroup swap0,
OOM events0; offset warnings0. The first generation's40 weight/mapping audit
rows passed; no completed response exists for the failing request.

Original failed run and controller are preserved remotely and locally in
`raw/dynamic-heldout-failure/qwen-dynamic-heldout-v2-failure.tar.gz`, SHA256
`5e87405d415b590372be5e3c24a91291ec11539fc84ce53a202d06c73b0fd9f0`.

The deployed `sglang/srt/utils/watchdog.py` checks scheduler feed counters;
layer migration progress alone does not feed it. Official reference inspected:
https://docs.sglang.io/docs/advanced_features/server_arguments
(`--watchdog-timeout`, default300, forward timeout terminates server).

v3 changes ONLY non-timed heldout watchdog to1800seconds and output identity.
It keeps finite timeout/resource protection, the complete128 requests in
original order, all hooks, threshold256/chunk512, model/mask/weights/C64 and
generation assertions. Restart from the first request; do not stitch32 old
responses into the new run. No new tolerance, threshold fitting or sample
selection occurs. Active CMMLU v2 and all original candidate bytes remain
untouched. Runtime implementation is unchanged.

`run_qwen_dynamic_heldout_retry.py` verifies the immutable candidate/controller
and adds a fail-closed attempt adapter. Local tests verify config changes are
restricted to watchdog/output paths and reject an existing watchdog override.
Observed red2fail then full28test PASS. v3 actual success remains pending.

IMPORTANT final-release integration: original audit/performance scripts point
to heldout-v2, which must REMAIN FAILED. Before performance release, add an
explicit audited v3 attempt-path adapter; do not overwrite/symlink v2 evidence
or change the old candidate. Audit and hash the v3 retry manifest/ack as well
as all raw responses. Performance timeout is NOT changed by this retry.

## 15:46 UTC follow-up

The exact failed request holdout_032 now returnedHTTP200,880input tokens,
16output tokens,80layer updates,482.895seconds. This verifies elimination of
the300second timeout for the observed failure; fullheldout qualification pending.

Explicit versioned audit/release adapter implemented in
`tools/qwen_dynamic_retry_release.py`. It checks retry provenance, preserves
FAILEDv2, reuses all original audit assertions and changes only attempt paths.
Performance entrypoint re-runs audit and validates exact summary/retry hashes
before invoking unchanged timing/isolation protocol. It does not write approval.
New negative regression rejects wrong attempt, incomplete/failed gates and
wrong retry digest. Local29testsPASS.
