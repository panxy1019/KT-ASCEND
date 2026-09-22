# Q64-C64 initial full-model bring-up — 2026-09-19

Actual execution on `.22`, owned `kt-round8-qwen35-prep`, physical NPU2;
CPU128–191, NUMA4/5, 64workers split32+32; memory384GiB. Historical A3
results are not substituted for this 910B execution.

Frozen model Qwen/Qwen3.5-35B-A3B revision
59d61f3ce65a6d9863b86d2e96597125219dc754, full40-layer CPU conversion
manifest35ca0ac63a3646ba7b89675053bbead19724325fa9d73f99910ef3c156a84405.
Q64 is64/256 NPU resident experts per layer, top8 routed, not an all-NPU oracle.

Launch-v5 loaded14/14 checkpoint shards and all40 CPU expert wrappers. Server
ready01:02:50 UTC; first diagnostic request completed01:04:28. It returned
8 tokens: `我是 Qwen3.5，阿里巴巴`. All40 structure checks passed: BF16 resident
weight dim0=64, frozen logical mapping/mask, Deferred0, finite MLP outputs.
Captured CPU route assignments5932; NPU2068. These are actual Hybrid executions,
not only server-import success. Shared presence is not yet exactly-once proof;
separate operand composition oracle is queued for clean restart.

The first response includes cold compilation; no throughput claim. Hooks sync
and copy diagnostic data and must be removed from timed performance.

Raw immutable initial evidence archive `q64-initial-bringup-evidence-v1.tar.gz`
SHA256170637557d81e3db0d8ad862398e5fa9403753e7ef13654bb1ff5b92ba657313.
Includes actual source/native hashes, mask, command, admission, request/response,
structure results, and three original failed launch logs. Do not overwrite.

Status: **SHORT_Q64_BRINGUP_PASS_NOT_QUALIFICATION**. Four-prompt repeated/prefix
matrix running; clean restart, shared composition, full32-prompt generation,
quality and performance remain separate gates. No Q32 promotion yet.

## Process A compact completed

32/32 requests across four distinct prompts (Chinese, English, math, JSON),
six16-token repeats, one32-token baseline and one8-token prefix per prompt:
all exact and finite. This is NOT the later32-distinct-prompt screen.
Protocol SHA65313739d00574db82563267c9a530279b6576d2be11a645492cc1114b8ac277.
Full process-A archive SHA57b83a0ab823af335466559ae04593643112e71ecdd62c0491437d75088024f1.

Clean restart initiated after successful matrix. Docker stop reported delayed
exit event; subsequent inspect confirmed exited, OOMKilled=false, no processes.
Exit137 here follows intentional container stop, not a model/runtime failure.
Process B uses same production code and configuration plus first-forward shared
composition diagnostic; token comparison also screens observer impact.
