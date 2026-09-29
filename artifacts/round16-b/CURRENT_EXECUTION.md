# Round16-B current execution

Updated: 2026-09-29 11:31 UTC.

Status: `B16_FULL_RUNTIME_AND_REQUEST_MEASURED_NO_PROMOTION`.

The former scaffold and `live_eligible=false` constructor block are resolved.
The exact vLLM-Ascend image now constructs only physical Q32 routed-expert
weights while keeping 512 logical experts, selectively loads resident rows,
executes real mixed C128/MXFP4 plus BF16/NZ29 MoE, keeps all 48 layer owners
resident, and runs a TP1 eager full model.

The completed stages and canonical evidence are:

- V1: `REAL_IMAGE_Q32_CONSTRUCTOR_VERIFIED` in
  `evidence/20260929-round16-b-q32-v1-device-v003/receipt.json`.
- V2: `REAL_Q32_LOADER_AND_LAYOUT_VERIFIED` in
  `evidence/20260929-round16-b-q32-v2-device-v007/receipt.json`.
- V3: `REAL_HYBRID_MOE_VERIFIED` in
  `evidence/20260929-round16-b-q32-v3-device-v007/receipt.json`.
- V4: `ALL_LAYER_RESIDENCY_VERIFIED` in
  `evidence/20260929-round16-b-q32-v4-device-v001/receipt.json`.
- V5: `PASS_ROUND16_B_V5_FULL_MODEL_SMOKE` in
  `evidence/20260929-round16-b-v5-smoke-v023/result.json`.
- Request comparison: `PASS_ROUND16_STRICT_REQUEST_COMPARISON`, disposition
  `B16_MEASURED_NO_PROMOTION`, in `ROUND16_REQUEST_COMPARISON.json`.

The KV fix is deliberately not described as a single-page `unsqueeze(0)`
solution. The device component evidence uses three physical pages, a
non-identity block table, two KV heads, 24 query heads, exact cache writes and
an independent sparse-GQA attention reference. Full-model qualification is
limited to cumulative context 256; full-model long-context cross-page
generation remains outside this round.

Four profiler-OFF request blocks ran as `S0-A -> B-A -> B-B -> S0-B`. For W2,
B/S0 TPOT ratios were `1.22502` and `1.19442`, while serial throughput ratios
were `0.82757` and `0.85031`. B is therefore not promoted. The default remains
S0 SGLang + KT.

The B engine ran eager with graph mode NONE. Native vLLM-Ascend mainline
operators remained in use outside the KT expert bridge, but no graph replay
coverage is claimed. B's log reports 24.2668 GB model weights, 2.85 GiB peak
activation, 23.04 GiB KV cache, and 47.34 GiB torch reserved after warmup. A
Host-cgroup peak was not captured by the B wrapper and remains UNKNOWN.

All B model processes exited and were removed. Formal cache retirement ran
only after archival and hashing and used registered read-only model/bank files
with no active references. No external workload was modified.
