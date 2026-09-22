# Round9 KT/Qwen mainline attribution — bounded execution completed

Run `round9-kt-mainline-20260922`. Qualification remains PENDING. Data below are bounded probes, not formal performance results.

## Verified baseline and numerical contract

Frozen source203files and overlay2323files matched. Baseline full commit57de93dfbb5775f1265378c1eae737edb8d5cc3e. Native extension SHA614ae60f70ff70d1e47d7fd32e857f887f874b6248eb332172c8a637cb471a6d. Torch2.9.0+cpu, torch_npu2.9.0.post2, kt-kernel0.6.1. Bank identity uses trusted complete receipts plus live metadata checks; the full checkpoint was not rehashed in this round. Layer0 GGUF was freshly hashed by the independent replay and matched e9d6b27f352d501a9ed3737ec5d3ab140b9d1f5bda22d2e36ef9ef0bcc45b08b.

Actual config:48layers, hidden2560,512experts/top10;36GDN and12QSA. `ple_layer_ids=[2]` in model config is the model's one-based second-layer placement (runtime layer_id1). Physical15 shared contract, no active training NPU PID during all observed samples. No inference of quiet Host/DRAM.

## Workload observations

| Probe | Actual input/output | Output IDs | Client wall / TTFT seconds |
|---|---|---|---|
| v62 base OFF |5 /1|271|69.2337 / recorded in raw evidence|
| v62 chunk OFF |64 /4|6105,5983,13358,2438|288.1502 /282.8622|
| v63 base sampled |5 /1|271|68.3998 /68.3997|

These samples have loading/cache history and shared-environment effects. No speedup or formal overhead percentage is inferred.

## First sampled prefill

217 timeline events, no drops;13 completed NPU event duration pairs, no observer error. Sampled provider layers0 and3, not all48. The timeline interval spans65.0414s; client request wall68.3998s includes work outside that interval.

- Expert shadow prepare:65calls, cumulative Host-observed60.2026s;15,532,032,000 COPY_ACCOUNTED_BYTES.
- Shadow compute enqueue:65calls, cumulative Host0.0560s. This is enqueue overhead, not kernel compute duration.
- PLE gather/H2D dependency wait:0.000366s; consumer enqueue0.198311s, again not standalone kernel time.
- Sampled routed-expert event intervals:2117.14ms and1108.36ms. They include stream elapsed time across weight preparation/execution and cannot be relabeled pure GEMM time.
- Sampled GDN event interval61.483ms; QSA attention223.844ms includes its indexer subspan47.163ms. Do not add parent and child as disjoint work.

The strongest current lead is expert preparation cost. This does not prove disk, DRAM, DMA or NZ conversion individually dominates. CPU packing/GEMM split and DRAM bandwidth remain UNKNOWN; absolute cross-stream overlap is uncalibrated.

Normal capture, replay and cancellation regression results follow.

## Completed normal capture window

v63 normal64-input/4-output matched all baseline token IDs, client wall278.5664s. The ten-input/eight-output capture completed in92.3606s. Across these three sampled requests there are14batches (4extend,10decode),242completed NPU event duration pairs,190sampled decode spans, no observer errors or timeline drops. This remains PARTIAL_ATTRIBUTION because native worker execution and clock-calibrated cross-stream dependencies are absent.

Normal64-token profile:274shadow prepare calls, cumulative Host264.2664s,75,959,500,800 COPY_ACCOUNTED_BYTES. These durations may include Host work, format preparation and transfer waits; they are not a direct DMA bandwidth measurement.

Decode sampled layer0/3 CPU submit, result-wait and resident calls are present. The first sampled routed-expert parent has an834.198ms stream interval, while subsequent sampled parents are about1.919–2.505ms. Inner submit/wait spans do not explain that first-parent cost, so it is retained as unattributed initialization/other work, with no asserted cause. QSA attention includes indexer time and cannot be summed with it.

Captured real row: hidden/input and output[1,2560]BF16; topk IDs[1,10]INT32; weights[1,10]FP32. Routes[20,150,14,111,203,414,211,358,33,230]. Actual resident0–31 means two BF16 NPU and eight MXFP4 CPU assignments. Both actual weight tensors areNZ29. Independent replay passed; details follow.

## Independent real-input replay

v20 exited0 with PASS_REAL_LAYER0_DECODE_REPLAY. One true layer0 decode row was replayed three times through the actual KTEP path, with128 CPU workers,32 original-BF16 NZ29 NPU experts and the MXFP4 CPU bank. The replay used the captured contiguous0–31 residency, not the legacy synthetic stride16 mask.

| Comparison | Relative L2 error |
|---|---:|
| CPU partial vs MXFP4-bank oracle |0.000117252872|
| NPU partial vs original BF16 oracle |0.000000088174|
| Routed merge vs mixed oracle |0.000305562251|
| Replay vs full-model captured output |0.0|

All three output SHA256 values equal0851f485287fca4f258ac946482d32d807e3220dfa5ace0398860198f5dd7897. The existing exact F32-partial→BF16 merge assertion passed. All64 resident projections matched canonical BF16. This proves one real mixed-path component fixture, not general model quality or all route shapes. The314,572,800bytes of resident weight tensors are actual32-slot storage.

## Cancellation regression and response limitation

PASS_MIDPREFILL_CANCEL_AND_REUSE.181input tokens were requested. Abort was sent when chunk[start32,count32] began. Only chunks0–31 and32–63 completed; chunk64 was not submitted. The terminal response was abort with0output tokens. CANCEL_SEEN was followed by SUBMIT_FENCE_PUBLISHED, REQUEST_RETIRED and CANCEL_CLEANUP_COMMITTED. The same process then answered the5-input/1-output reuse request with token271.

CONTROL_SENT→CANCEL_SEEN took339.116508971s. CANCEL_SEEN→cleanup took0.040032246s. Cancellation still waits for the in-flight chunk before the scheduler accepts it; this is a material responsiveness limitation, not a low-latency cancellation result. The trace supports no new chunk after acceptance in this tested interleaving; it does not establish a universal deadline or every failure interleaving.

## Resource and environment closure

| Run | Max board MiB | Max owned process MiB | Max training process MiB |
|---|---:|---:|---:|
| v62 OFF |29317|26550|0|
| v63 sampled + cancel/reuse |29316|26550|0|
| v20 replay |3882|1060|0|

No STOP_OWNED guard action occurred; monitors ended with owned exit. Full-model processes were explicitly stopped after probes; replay exited0. Final inspection found no running Round9 containers, while the admitted training container remained running. Its NPU process usage was zero throughout these samples, so active training contention remains unqualified. Resource snapshots and full remote monitor hashes are in evidence/resource-summary.json; final npu-smi and ownership state in evidence/cleanup.json. NUMA startup admission does not reserve future free memory; process RSS and file cache are distinct. Scoped project-file DONTNEED was used only after owned processes stopped; file content identities remained unchanged.

Used2/2model starts,7/8HTTP requests,3/3device-profile requests,1/2component starts. No automatic retry, formal performance, CMMLU, long context, training changes, Round7 timer changes or dependency upgrade. No push/merge/tag.

## Implementation, verification and remaining uncertainty

Default-OFF observation wraps sampled real providers without rebuilding native kernels. Bounded event pairs, request identity, real decode capture and a conservative offline summarizer were added. The replay entry point now accepts a validated real capture. Four review findings were fixed with RED→GREEN regressions; actual residency was corrected before the first physical replay.49 related local tests passed; frozen container host checks also passed. See PROBLEMS_AND_RESOLUTIONS.md for each issue.

The v63 observation build is immutable and differs from the subsequent replay snapshot in the real-mask validator/harness; both source manifests are retained. The running capture producer was correct, and was not hot patched. Source hashes for the delivered files are frozen in PATCH_MANIFEST.json.

Attribution status remains PARTIAL_ATTRIBUTION: native worker packing/GEMM and DRAM bandwidth are UNKNOWN, provider durations are inclusive, absolute cross-stream overlap is not calibrated. The final timeline includes the cancellation regression and reuse in addition to the three normal sampled requests; use v63-normal-summary.json for normal-workload conclusions. Cancellation traffic is not a performance workload. Token271 is a baseline consistency check; it is not a passed “answer OK” quality canary.

Decision: no behavior optimization selected this round. The concrete next experiment is bounded expert shadow preparation decomposition, specified in NARROW_EXPERIMENT_DECISION.md. PRODUCT_QUALITY, PERFORMANCE, real LONG_CONTEXT and DEPLOYMENT remain PENDING.
