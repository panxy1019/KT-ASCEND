# Concurrent source preparation — 2026-09-19

Latest user explicitly authorizes downloading/checking/preparing the next model
while Round8 remains active. This supersedes the earlier metadata-only limit.
It does not qualify Qwen3.8 or authorize interference with timed Round8 work.

## Source and transport

Official Hub API files_metadata at pinned revision produced144files, including
131safetensors, total360023351514bytes (~335.3GiB). `download_source.json` holds
size, LFS SHA256 or ordinary Git blob hash for each file. Source remains
Qwen/Qwen3.8-Flash-Next@de4b8e4d43b917e7706784d8bb445c9af86a3540.
Direct .22 huggingface.co443 refused; hf-mirror.com transport tested successfully.
A proxy probe was unsuccessful; no global host proxy/network settings changed.
Mirror files cannot pass solely on mirror metadata: verification compares the
independently obtained official manifest, then full index/header/shape/offset
coverage. Includes original source auxiliary tensors; conversion must later
exclude MTP/vision explicitly, not silently omit source shards.

## Admission and owned resources

Observed .22 filesystem free2877870014464bytes (~2.62TiB), host MemAvailable
551654316KiB (~526GiB). Historical swap~294MiB is recorded, not claimed zero.
Download admitted, full F32+PLE runtime NOT admitted.
Local desktop root only72MiB free: no large local downloads/cache expansion.

Remote `/home/admin/kt-round9-qwen38/models/<revision>` is a new isolated path.
Owned transient systemd service: `kt-round9-model-download.service`.
Four network workers with one-CPU quota,8GiB memory,swap0,Nice19,idleI/O,
write30MB/s and read50MB/s limits. Explicit filename allowlist, immutable
revision, no NPU. Partial files remain resumable; never claim completion based
on progress count or filename existence. Download success automatically enters
streaming hash verification and local safetensors header/index census.

Before timed Round8 performance set PAUSE_DOWNLOAD and verify the stopped
child group before timing. Guard additionally pauses for prep-container running,
unknown Docker state, host available<128GiB or diskfree<150GiB. Only owned child
PGID is signaled. This guard reduces risk; formal isolation still needs admission.

## Isolated environment preparation

New `kt-round9-qwen38-prep` container:
- image sha256:b81e664642aecd5f4cc8ec38b233aae38f65b9cf7feea40e54e13c08cd319347
- 2CPU,8GiB,swap0,networknone,devices[],nonprivileged,capabilities dropped
- read-only Round9 directory mount; no Round8 code/model mounts
- package inventory command exited0, container now stopped
- Python3.11.15,torch2.9.0,torch-npu2.9.0.post2,transformers5.3.0.dev0

The login environment prints inability to run npu-smi, expected with no device
access. Model config mentions transformers5.8.0.dev0; registration and serving
compatibility remain unresolved. This is PREPARATION_CONTAINER_READY, not
SERVING_READY. No installed host/Round8 packages were changed.

## Verification and next actions

Local tests:5download identity/census checks,29retained Round8 tooling checks
PASS. Negative tests cover same-size corruption, missing file, path escape and
index identity mismatch. Broader pytest suite couldn't start because local
Python has no pytest; no native/operator/A3 qualification claimed.

Remote live interlock tested: marker caused PAUSED and hf process stateTNsl;
after marker removal controller returnedRUNNING. Existing quality lanes kept
progressing (snapshot heldout67/128,CMMLU143/256), no restart performed.

Next: monitor byte progress/service failure; after complete audit131shards,
1658expected index tensors and actual dtypes/categories; freeze tokenizer,
template/license hashes; build explicit runtime compatibility patch plan.
Source download and validation may finish before Round8. Full F32 conversion
and device serving wait for Round8 closure, semantic audits and new resource
budget. Preserve all original Round8 and other users' workers.
