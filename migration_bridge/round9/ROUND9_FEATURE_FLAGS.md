# Round9 framework feature flags and identity contracts

Date: 2026-09-21  
Branch: `codex/round9-heterogeneous-working-set`

Every Framework Completion feature is opt-in and bound to immutable input files.
The launcher copies each selected profile into the evidence directory, records its
SHA256 in `payload-sha256.json` and `admission.json`, and the server verifies the
same digest before importing the feature. A missing flag means the preserved
baseline path is used.

| Plane | User-facing selector | Runtime identity | Supported bounded modes |
|---|---|---|---|
| Expert capacity and scheduling | `--framework-profile` | `ROUND9_FRAMEWORK_PROFILE` + `ROUND9_FRAMEWORK_PROFILE_SHA256` | Q64/Q32; Uniform; frozen Frequency; route collection |
| Dynamic boundary | `--placement` | `ROUND9_DYNAMIC_BOUNDARY_ENABLE` plus the candidate and device receipts | `static`, `request_boundary`; `prefill_end` through a Layerwise profile |
| Layerwise prefill | `prefill` in the framework profile | Frozen Layerwise device receipt | `LAYERWISE`, positive threshold, optional `update_residents` |
| PLE | `ple` in the framework profile | Frozen PLE device receipt | synchronous baseline when absent; exact-input `ASYNC` when present |
| Joint host budget | `joint_host_budget` in schema 4 | Frozen joint device receipt | Q32 + async PLE only |
| Context pool sizing | `--context-target` | `ROUND9_CONTEXT_PROBE_PROFILE` + SHA256 | 4096, 16384, 65536, 131072 derivation; actual 4096 allocation measured |
| QSA reference consumer | `--context-reference-profile` | `ROUND9_CONTEXT_REFERENCE_PROFILE` + SHA256 | `BF16_COMPACT`, `FP8_HOST_REFERENCE` |
| Streaming prefill | `--streaming-profile` | `ROUND9_STREAMING_PROFILE` + SHA256 | serial 32-token chunks with matching Layerwise, PLE and context profiles |

The supplied profiles are:

- `framework_uniform_q64_v1.json`: explicit Q64 Uniform baseline;
- the F1 generated Q64/Q32 frequency manifests: deterministic per-layer masks;
- `framework_ple_async_q64_v1.json`: Q64 Static plus PLE async;
- `framework_joint_q32_v1.json`: Q32 plus PLE async and shared host budget;
- `framework_prefill_end_ple_async_q32_v1.json`: Q32 Layerwise Prefill-End
  Dynamic plus exact-input PLE async for the F12 interaction cell;
- `framework_streaming_q32_v1.json`: Q32, Layerwise and PLE async for F11;
- `context_reference_bf16_compact_v1.json`: selected BF16 K/V rows through bounded staging;
- `context_reference_fp8_host_v1.json`: canonical host FP8 record decoded to BF16 staging;
- `streaming_bounded32_v1.json`: threshold 4 and physical chunk bound 32.

## Fail-closed rules

Profiles reject unknown fields, unsupported capacities, relative receipt paths,
missing or changed hashes and unqualified combinations. Frequency cannot be
silently combined with Layerwise or Dynamic. Joint host budgeting requires Q32
and async PLE. Streaming requires all three working-set planes at the same
threshold. Request, placement and allocation epochs are distinct identities;
stale, duplicate or incomplete ownership changes the affected runtime to
`BROKEN_RELOAD_REQUIRED` where rollback cannot be proven.

The feature-OFF v36 full-model regression produced the same output token IDs and
zero maximum output-logprob difference versus the corresponding PLE-enabled v35
smoke, and emitted no async PLE event file. This is the bounded default-OFF
regression; it is not a general numerical-equivalence claim for every profile.
