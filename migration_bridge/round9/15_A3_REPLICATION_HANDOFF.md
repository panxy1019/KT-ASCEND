# A3 verified model replication and development handoff

Date: 2026-09-20 UTC

The user authorized a post-download copy on `110.123.0.3` under
`/home/admin/models` for a possible Round9 development migration. This handoff
does not qualify the model or any A3 runtime path.

## Immutable source

- Repository: `Qwen/Qwen3.8-Flash-Next`
- Revision: `de4b8e4d43b917e7706784d8bb445c9af86a3540`
- Source host: `110.129.0.22`
- Source model: `/home/admin/kt-round9-qwen38/models/<revision>`
- Source gate: `download-state.json.status == DOWNLOAD_IDENTITY_PASS`
- Frozen source manifest: `/home/admin/kt-round9-qwen38/download_source.json`

The transfer cannot start from size completion alone. The source downloader must
first complete every manifest SHA and tensor census and publish the source gate.

## Destination and publication

- Host: `110.123.0.3`
- Staging: `/home/admin/models/Qwen3.8-Flash-Next/.<revision>.partial`
- Final: `/home/admin/models/Qwen3.8-Flash-Next/<revision>`
- Support: `/home/admin/models/Qwen3.8-Flash-Next/<revision>.round9-support`
- Service: `kt-round9-a3-replication-v1`

The service runs as `admin`, waits for the source gate, then uses direct private
network rsync with partial/append verification. It reruns the frozen-manifest
whole-file SHA verification and tensor census against the A3 staging directory.
Only a successful destination audit permits the staging directory to be renamed
to the final path. `replication-state.json` is the machine-readable state.

Admission inventory recorded about 1.03TB free on the destination `/home`
filesystem; the transfer requires at least 450GB free before it starts. No
existing `/home/admin/models` model is deleted or overwritten. A pre-existing
final path is a hard blocker.

## Development resources copied separately

The support directory contains these small, independently verifiable resources:

- `KT-ASCEND-round9-<commit>.tar.gz`: tracked Round9 plan, tools, tests and evidence.
- `KT-ASCEND-round9-<commit>.bundle`: local Git branch history for reconstruction.
- SHA256 sidecars for both artifacts.
- `download_source.json`, `verify_download.py`, destination verification receipt,
  and the replication state written by the watcher.

The Git bundle records the superproject. Submodule object databases are not
embedded; their recorded gitlinks must be fetched from their respective sources.
The current local gitlinks are:

- `third_party/custom_flashinfer`: `fd94393fb5b8ba8bae9c0bd6ab1c2a429d81ac76`
- `third_party/llama.cpp`: `a94e6ff8774b7c9f950d9545baf0ce35e8d1ed2f`
- `third_party/pybind11`: `bb05e0810b87e74709d9f4c4545f1f57a1b386f5`
- `third_party/sglang`: `f02f304a25364d1e8fe9c52515a22f20f5089a52`

Container images, compiled CPU extensions, Ascend wheels and future converted
expert artifacts are not yet frozen Round9 deliverables. They must be added only
after R9-A chooses and verifies the actual runtime combination. Copying an older
Round8 image now would not prove Qwen3.8 compatibility.

## Operational checks

```bash
systemctl status kt-round9-a3-replication-v1 --no-pager
cat /home/admin/models/Qwen3.8-Flash-Next/<revision>.round9-support/replication-state.json
```

Valid terminal success is `REPLICATION_IDENTITY_PASS`. Until then, the A3 model
is `PREPARED` or `TRANSFERRING`, never a qualified or runnable Round9 baseline.
