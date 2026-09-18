# Qwen import and sample acquisition — 2026-09-18

## Latest acquisition recovery

19:30 checkpoint: v3 also exited on remote partial-body closure; cache grew to
7.6GiB, demonstrating useful resume rather than a full restart. Launched v4
bounded three-attempt same-command resume loop, single worker, unchanged
revision/cache/proxy/timeout. Log `sample-download-proxy-v4.log`. Do not start
another downloader until its owned driver exits; neither incomplete cache size
nor CLI completion substitutes for whole-shard SHA validation.

19:14 checkpoint: v2 terminated with RemoteProtocolError (partial body after
1,314,145,305 bytes of one shard), retained total2.5GiB incomplete cache.
Resumed SAME filenames/revision/directory with max-workers1 and
HF_HUB_DOWNLOAD_TIMEOUT=120, PID1505716, log `sample-download-proxy-v3.log`.
Tunnel79305/port27892 remains required. No force-download or cache deletion.
Download/hash validation is still pending. Four local verifier rejection tests
cover wrong file size, SHA, header length and dtype; these are not weight tests.

Direct download PID1323468 exited1 with Network unreachable; v1 log preserved.
Local workstation has only1.3GiB free, so do not stage10.7GB there. A loopback-only
SSH reverse proxy through the existing A3 jump now routes .22:27892 to local
127.0.0.1:7890. HTTPS status200 verified; no global proxy configuration changed.
First attempted port17890 was occupied and NOT modified. Active tunnel tool
session79305; preserve it during download, verify/reestablish if interrupted.
Host HF download restarted PID1415742 using per-process HTTP_PROXY/HTTPS_PROXY
http://127.0.0.1:27892 and HF_HUB_DISABLE_XET=1. Same files/revision/path.
New log `sample-download-proxy-v2.log`. Last observed201MiB downloaded and
process active; no checksum PASS yet. Do not duplicate healthy acquisition.
Read-only verifier `tools/inspect_sample_shards.py` prepared locally; after both
files complete, deploy separately and verify whole-shard SHA before inspecting
Layer0 packed shape/dtype/offsets. This tool is not yet remote-executed.

Actual read-only reference environment successfully imports
`sglang.srt.models.qwen3_5` from
`/workspace/kt-src/third_party/sglang/python/sglang/srt/models/qwen3_5.py`.
This establishes import compatibility only, not construction/GDN execution.
Remote log `/home/admin/kt-round8-qwen35/qwen-import-v1.log`, SHA256
`51e7bd24d41848c61fffe2e7938068b8c10a70e8663d292aa6beb919d8bb8ae6`.
No source or shared package edits. An independent dependency build is pending.

Host capacity snapshot: MemAvailable 986GiB, disk available 2.9TiB. Host swap
contains historical 43MiB; do not label swap zero or infer active swap I/O.
Only model import ran; no model loading or timed performance.

Frozen index assigns Layer0 text expert packed gate_up to shard6 and down to
shard12. MTP entries excluded. Started host `hf`1.14.0 acquisition of ONLY these
two shards at revision `59d61f3ce65a6d9863b86d2e96597125219dc754`:

```text
hf download Qwen/Qwen3.5-35B-A3B model.safetensors-00006-of-00014.safetensors model.safetensors-00012-of-00014.safetensors --revision 59d61f3ce65a6d9863b86d2e96597125219dc754 --local-dir /home/admin/kt-round8-qwen35/model-sample --max-workers 2
```

Background PID at launch:1323468 (verify identity, do not trust stale PID).
Log `/home/admin/kt-round8-qwen35/sample-download-v1.log`.
Do not duplicate a healthy download. Acquisition is NOT yet checksum-verified.

Expected identities from previously frozen Hub manifest:

| Shard | Bytes | SHA256 |
|---|---:|---|
| 00006 | 5368709816 | af5f17d292df6b200ee928d7dac30a66eab3e8f6d5d88a84fe1d27859d4a64fe |
| 00012 | 5368710456 | d0f0bd7a5f09e6e05aee9a6ef5ba97fe678b0e19b891a81969682b0bdc9bb5c7 |

After completion: verify size and whole-file hashes, inspect actual packed tensor
shapes, extract Layer0/expert0 without MTP, prove BF16→F32 value identity, then
same-operand CPU/reference probe. Do not start full conversion before sample PASS.
Full weights, GDN forward and Q64/Q32 remain unqualified.
