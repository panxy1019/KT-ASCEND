# Real-layer multi-expert diagnostic entry

2026-09-19 00:20 UTC, after full checkpoint identity PASS. New tool
`tools/multi_expert_probe.py` verifies relevant source shard hashes again,
converts one real layer's256 experts to F32 GGUF, and requires exact GGUF
readback. Only one layer is materialized per process; no full120GiB conversion.

Then execute eight fixed global IDs `[0,1,31,63,127,191,254,255]`, qlen1/8/64,
ten repeats each, finite/exact repeats and local relative-L2<=0.01 against a
BF16-boundary CPU reference with route weight applied once. CPU64/two32
subpools/NUMA4,5 unchanged. All experts CPU-owned for this isolated operator
diagnostic: NOT the eventual Q64 or Q32 placement. No router/shared/cache claim.

Layer0 launched in `kt-round8-qwen35-prep`, driverPID3034422 (verify identity),
log `/home/admin/kt-round8-qwen35/multi-expert-layer0-v1.log`, output directory
same stem without.log. Initial import warning `can not use command: npu-smi info`
matches earlier nonfatal probes. Result pending; preserve healthy execution.
Do not count local syntax checks or existing12 unit tests as A3 operator PASS.

Next: inspect result/failure, preserve raw evidence, repair actual defects only;
after Layer0 success repeat early/middle/late coverage in separate processes.
Full-model source/overlay reconciliation and model/cache integration remain
required before Q64→Q32. No Dynamic/Deferred/Graph and no push/tag/main merge.
