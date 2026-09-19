# Bounded alternate transport pilot — 2026-09-19 22:39 UTC

HF/httpx failed repeatedly with TLS bad-record-MAC, including a single-worker
attempt. Official direct HTTPS remains unreachable from .22. A TLS-verified
systemcurl 1MiB range succeeded against the already authorized mirror. Exact
network/TLS root cause is not established; the hypothesis is that a separate
client using short bounded range transfers can recover one shard safely.

Do not change source revision, disable certificates, replace source weights,
remove HF incomplete cache, or claim mirror delivery proves model identity.

## Frozen pilot

- Local code commit before dispatch:2cdee26d0.
- Service:kt-round9-curl-pilot-v1 on .22, started22:38:40UTC.
- Directory:/home/admin/kt-round9-qwen38/curl-shard-pilot-v1.
- Source revision:de4b8e4d43b917e7706784d8bb445c9af86a3540.
- File:model-00029-of-00131.safetensors,3200016032bytes.
- Official SHA256:40d2f2291ff4800373087cd00f1f95c3bd69331c32f39cfdd5d43516764b03ec.
- Serial64MiB chunks;180s request cap;20MiB/s network cap;no blanket retry.
- CPU100%,memory8GiB,swap0,nice19,idleIO,read50MB/s,write30MB/s.
- Uses existing guard before/during transfers and final checksum subprocess;
  PAUSE_DOWNLOAD and activeRound8prep continue to stop the owned child group.
- No NPU, no production model directory publication, no existing cache writes.

Each chunk requires HTTP206, exact Content-Range start/end/total and exact body
length before append. Headers/failure evidence retained; only copied owned chunk
temporary removed after durable append. Final file must independently pass the
existing official-manifest size/SHA verifier. Successful pilot is not whole-model
DOWNLOAD_IDENTITY_PASS. It stops afterONEshard; audit before broader adoption.

## Verification

Range validation tests first failed on omitted validation, then passed for
redirected206 and rejection of ignored/wrong-range/wrong-length/wrong-total.
Round9 tooling suite8/8PASS. Device transfer and fullSHA result pending at launch.
Installed curl7.81.0 supports the used options; TLS remains enabled and redirects
restricted toHTTPS. Primary reference:https://curl.se/docs/manpage.html.
The manual warns blanket retry can duplicate or alter partial-output handling;
this pilot instead preserves first failure and never retries implicitly.

Monitor progress.json, download-state.json and owned service journal. Old
kt-round9-model-download remainsFAILED; do not run it concurrently with pilot.
If pilot fails, preserve it and investigate, not loop new downloads indefinitely.
