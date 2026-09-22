# B0 final result and independent study freeze

Evidence state: target-executed on110.129.0.22, dedicated `kt-r65-static-offload`, Ascend910B3/NPU0. Do not confuse this with the historical A3 hardware. Existing installed production extension and91 runtime source files remain unchanged.

## Backend decomposition

`BACKEND_DECOMPOSITION_GATE_PASS`:128 CPU-owned expert instances,384 real decode cases (three per expert),224 prefill cases across qlen11/16/64/128;608 total. Some decode expert/input pairings are explicit unrouted controls, not a claim of natural routing coverage. Production and NPU ten-repeat hashes are exact; all values finite. Native diagnostic stage capture matches production outputs608/608, and one-slot vs64-slot NPU layout matches608/608.

| Comparison | Maximum relative L2 |
|---|---:|
| LLAMAFILE vs canonical FP32 |0.004483591392794621|
| LLAMAFILE vs NPU |0.0015245552599669338|
| NPU vs canonical FP32 (down) |0.004482310916015284|

All measured local stage envelopes remain below0.01. No material extra LLAMAFILE numerical defect was demonstrated. This is a bounded operator conclusion, not proof that all model trajectories or quality are equivalent. No production arithmetic rewrite, replacement GEMM or NPU-for-CPU path is adopted.

The diagnostic extension lives in an isolated package. Model studies use the unchanged installed production extension. Prior failed diagnostic capture/build/capacity attempts remain preserved and are not counted as production failures or qualifying runs; successful reruns use distinct evidence paths.

## Frozen independent study

| Artifact | SHA256 |
|---|---|
| QD6 |b70449cc6f744307458c21d4463c8496c5ba09e60aa21e686831b3217cf29570|
| HD6 |e812852ccd9ca9f9427b31f56c29ebca8167c7b7116ecd27b8c8e2aa973ab15f|
| Quality |9618d054cf765e1d6328d1581435ec97f9f4271715b794dc07e0a9889a483fcb|
| Contract V1 |4a374b76690ccbafc4fb26006463eb416e80b06a3b834992595e67365393c79b|
| Freeze manifest |de0452fc77de0b2cc11f5f9bf31992e3f9f2aba1e8a893bb5f100173dad4b68c|

QD6 and HD6 each contain12 new prompts and64 target positions per prompt. Additional lookahead forwards are audited, not silently discarded. Quality is an independent32 GSM8K/256 CMMLU engineering screen, with no observed net accuracy loss/no invalid-count increase required; it is not population noninferiority and does not inherit Deferred's2pp tolerance.

QD6 starts with the All-NPU oracle, then four-R32 on identical histories. HD6 cannot start until QD6 integrity/system audits and candidate freeze are complete. New contract treats model distributions as explicit diagnostics; it does not refit the historical failed global maximum. Historical Round6.5 remains BLOCKED. Round7/Deferred/all-MoE promotion remain unqualified.
