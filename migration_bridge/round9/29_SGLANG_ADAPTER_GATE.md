# Round9 SGLang config, pools, metadata and adapter gate

2026-09-20, 110.123.0.3 physical15. Bounded actual SGLang modules,
original weights and synthetic activations. Full model remains NOT_RUN.

## Source and results

Base f02f304a25364d1e8fe9c52515a22f20f5089a52; semantic donor
4ccff141dbe992794f9da6c3aa23535b4f72000d. Both patches applied sequentially
to a fresh f02 archive. sglang_patches/manifest.json binds patches, overlays
and canonical modules; tested-source-sha256.json binds actual remote files.

- Config import: 48 layers, hidden2560, 512 experts/top10, GR4/lowrank320,
  PLE layer1/state10240x9/history2, 12 full-attention and 36 GDN layers.
- Actual model meta construction: 1091 parameter tensors; all36 A_log FP32.
  This is metadata, not real full-model loading.
- Loader metadata v4: all1658 checkpoint keys mapped, 1094 destinations and
  128 PLE shards accounted. Small hash buffers actually loaded. Missing Q
  projection and shifted PLE shard boundaries rejected.
- Pool boundary: 8 tests PASS, including lifecycle, budget, invalid profiles,
  final legal physical token/group, actual NPU paged KV flatten/token addressing.
- Adapter v4: 36 runs = four modules x three paths x three repeats. Paths:
  whole17, chunks[3,4,10], token decode. Every run frees/reuses the real request
  and Mamba slot. Initial GDN/PLE states are checked; same-path hashes repeat.
- GDN layer0 and QSA indexer layer3 cross-path relative errors zero;
  PLE layer1 maximum 0.00005635638. Cross-path output/state gate0.01.
- Actual model self_attention includes packed QKV, Gemma norms, native RoPE,
  paged KV, compressed index, sparse GQA, sigmoid output gate and output
  projection. Independent FP64 oracle maximum relative L2: 0.00404127131,
  below the frozen0.03 gate.
- Peak NPU allocated memory370674688 bytes, container normal exit. No full
  model budget is extrapolated from this bounded peak.

Attention fixture replaces only the unused MoE constructor with Identity and
calls self_attention directly. It does not qualify a complete decoder. Token
mapping is explicitly a fixture, not scheduler allocation qualification.
Actual KTEP Q64/C64 is separately qualified in reports27/28.

## Failures and repairs

- Config donor default constructor assumed rope_scaling exists under TF5.3;
  access is now optional, preserving loaded checkpoint rope_parameters.
- f02 language_only selects encoder disaggregation. Fixture instead uses
  enable_multimodal=False with the actual text-only model class.
- Meta tensor item() cannot audit PLE values: fixture materializes three tiny
  hash buffers only.
- PLE logical rows320001446 pad to storage320001536 under alignment128.
  Production loader enforces total padding and each of128 shard boundaries.
- Adapter v1 treated shared GDN attn.A_log alias as a second checkpoint key:
  unique named_parameters now matches the actual model loader.
- Adapter v2 missed f02 ForwardBatch.seq_lens_sum: fixture supplies cumulative
  sequence length. v3 three-adapter and v4 four-adapter gates pass.
- Unsupported memory saver/offload/disaggregation modes explicitly rejected;
  pool documentation no longer claims unwired host offload support.

Evidence: evidence/physical15/20260920-sglang-pools; failures, metadata receipts,
adapter-v4.json, exact tested probe, container configs, source hashes and
SHA256SUMS retained. All copied artifact hashes verified locally.

## Remaining

Real full-model loading, runner PLE table binding, shared expert/router boundary,
final GR mixer, scheduler execution and full-model resource admission remain.
Concurrent training envelope unresolved; idle telemetry is not a reservation.
Both timers PAUSED. No full-model PASS, generation or CMMLU claim.

## Formal loader PLE binding follow-up

The real f02 `_initialize_model` entry, rather than direct construction, first
reproduced `real model loader did not bind original PLE table` after successful
metadata loading. Patch0003 passes the original local model_path to Qwen4.
After all load audits succeed, the model owns a CheckpointPLETable, validates
its row count against the audited padded vocabulary and binds every PLE layer.
It closes any previous owned reader on replacement. Reads remain synchronous,
BF16 and bounded to requested rows; no complete PLE table is allocated.

The green actual initialization/metadata audit binds and reads first/last logical
rows. Missing Q and shifted PLE shard negatives still reject. Adapter v5 replaces
the table fixture with the production reader: all36 adapter runs PASS, exit0.
All three patches apply to a fresh f02 archive. Evidence retains red/green logs,
receipts, exact revised probes, container states and updated source hashes.
This closes table binding, not actual full-model materialization or admission.
