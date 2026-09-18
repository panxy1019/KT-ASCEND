# Convolution state and full checkpoint acquisition — 2026-09-18

Real deployed causal_conv1d NPU API exercised with frozen Qwen dimensions:
8192 channels, width4, BF16 synthetic inputs/weights, seed123456. Prefill64
then4 continuous decode steps. All finite; prefill and each decode retained
history state is exact against input history. Relative-L2 to CPU reference:
0.00245851,0.00311547,0.00305565,0.00300658,0.00304463.
Raw result `conv-state-v1.json` copied locally. This is a primitive/state
diagnostic, not full serving cache/model qualification. No production edits.

Host snapshot at next acquisition:986GiB available RAM,2.9TiB free disk;
historical host swap48MiB (not zero). No model server launched, no claim of
full-model memory admission from this disk-acquisition snapshot.

Started full checkpoint acquisition under SAME frozen revision and directory:
`hf download Qwen/Qwen3.5-35B-A3B --include '*.safetensors' '*.json'
--revision 59d61f3ce65a6d9863b86d2e96597125219dc754
--local-dir /home/admin/kt-round8-qwen35/model-sample --max-workers 1`.
The misleading historical directory name is retained to reuse verified shards;
it will contain the full checkpoint when complete. No alternative model/version.
Transport uses scoped proxy27892/tunnel79305, download timeout120, XET disabled,
bounded6-attempt resume loop. Owned driverPID2166113; verify identity before any
action. Log `/home/admin/kt-round8-qwen35/full-checkpoint-download-v1.log`.
Preserve healthy acquisition. Require all14 shard hashes plus frozen metadata
checks before model loading. CLI completion is not artifact qualification.

No full120GiB F32 conversion started. Pending: multi-expert/top8 integration,
early/mid/late operands, independent dependency lock/build, actual model/cache
construction and Q64→Q32. Completed primitive tests must not be substituted.
