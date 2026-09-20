# A3 development resource manifest

Date: 2026-09-20 UTC

Purpose: define the additional `.22 -> .3` handoff needed to continue Round9
without copying obsolete models, transient download experiments, or unqualified
runtime binaries. This document does not qualify Qwen3.8 execution on A3.

## Destination

- Host: `110.123.0.3`
- Model root: `/home/admin/models/Qwen3.8-Flash-Next`
- Revision: `de4b8e4d43b917e7706784d8bb445c9af86a3540`
- Support root: `/home/admin/models/Qwen3.8-Flash-Next/<revision>.round9-support`
- Model publication gate: `replication-state.json.status == REPLICATION_IDENTITY_PASS`

The model is copied into a hidden partial directory, fully reverified on A3, and
only then atomically published. The transfer/verification service continues in
the background; a partial directory is never a runnable or qualified model.

## Copied reconstruction artifacts

| Artifact | SHA256 | Purpose |
|---|---|---|
| `KT-ASCEND-round9-f1828a009.tar.gz` | `85519be451ce2a2e98b084b9fd5ef5fff708fab337025e22b96990d840fc4b4e` | Tracked Round9 plans, tools, tests, evidence, and handoff documents |
| `KT-ASCEND-round9-f1828a009.bundle` | `111eecb914212a394456e705f96c1e7d8c83ea4b2d5773eb272a2df525ab1c7d` | Complete local superproject history through `f1828a0096420dcb31b79ebb6c1fe4f808865973` |
| `round8-handoff/KT-ASCEND-round8-f1828a009-handoff.tar.gz` | `ff1208083b0bbf7e42bf632ea3cccbd329e3ef61b2a06e19d831f5b51829d188` | Round8 final reports, frozen handoff manifests, Dynamic evidence, unit tests, and evaluation tools |

The Git bundle has been verified by importing it into a temporary repository.
The Round8 archive and both Round9 artifacts pass their destination SHA files.

## Copied source-identity evidence

The `source-final/` directory preserves the immutable `.22` download result:

- `download-state.json`
- `download_source.json`
- `verify_download.py`
- `evidence/verification-1789881848096529758.json`
- `SHA256SUMS`

The source receipt reports `PASS` for repository
`Qwen/Qwen3.8-Flash-Next`, revision
`de4b8e4d43b917e7706784d8bb445c9af86a3540`. A second, independent destination
receipt will be produced under `destination-verification/` before publication.

## Submodules

The superproject records these exact gitlinks:

- `third_party/custom_flashinfer`: `fd94393fb5b8ba8bae9c0bd6ab1c2a429d81ac76`
- `third_party/llama.cpp`: `a94e6ff8774b7c9f950d9545baf0ce35e8d1ed2f`
- `third_party/pybind11`: `bb05e0810b87e74709d9f4c4545f1f57a1b386f5`
- `third_party/sglang`: `f02f304a25364d1e8fe9c52515a22f20f5089a52`

The current local worktree has uninitialized submodule directories, so no
unverifiable source-tree snapshot was fabricated. Rebuild must fetch these exact
objects from their recorded remotes, then verify the gitlinks before compiling.

## Deliberately not copied

The following data is not required to continue Round9 and remains on `.22`:

- About 199 GB of Qwen3.5/Round8 model, F32 expert pool, and sample conversions.
  The compact frozen Round8 evidence package above is the migration reference.
- `range-recovery-v1` staging/failure data and all discarded throughput pilots.
  Their accepted final manifest and verification receipt are preserved instead.
- Triton caches, page cache, Python caches, logs, and temporary tensors.
- Image `kt-r67-frozen-runtime:20260917` (source digest
  `sha256:b81e664642aecd5f4cc8ec38b233aae38f65b9cf7feea40e54e13c08cd319347`).
  It is a Round6/7 reference image, is absent on A3, and is not yet a qualified
  Qwen3.8 runtime. Export it only if R9-A explicitly selects it as the base.
- Compiled CPU extensions, Ascend wheels, overlays, converted Qwen3.8 CPU expert
  banks, and PLE artifacts that do not yet exist as frozen Round9 deliverables.

This exclusion prevents an old environment or a large historical model from
being mistaken for a qualified Qwen3.8 input.

## Automatic continuation

Service `kt-round9-a3-replication-v1` remains the single owner of the model
transfer. It resumes partial files, performs full destination SHA/tensor census,
publishes atomically, and records a terminal state. Do not start another rsync
against the same partial directory.

After `REPLICATION_IDENTITY_PASS`, the next session must:

1. verify all support `SHA256SUMS` files and the destination receipt;
2. reconstruct or clone the superproject and checkout the four exact gitlinks;
3. run R9-A source/runtime/resource admission on A3;
4. freeze the actual container image, wheels, CPU operator build, and overlays;
5. only then add those runtime artifacts to this manifest.

Until step 4 is complete, `MODEL_EXECUTION=NOT_RUN` and no A3 runtime is claimed.
