# Dynamic DEV entry audit — 2026-09-19 13:40 UTC

Status: **NOT READY FOR HELD-OUT OR CMMLU**. This is an audit finding, not
a claim that the qualification audit has closed. No production change made.

## Verified evidence

Independent re-reading of raw generation events for DEV256 and DEV512 confirms
120 updates each, 40 layers, generations 1/2/3, identical repeated-A selections,
32 distinct resident IDs, and recorded exact weight/mapping/mask assertions.
All 40 layers change resident selection for workload B. Both candidates actually
updated on 512-token forwards. Parallel diagnostic latency is not performance.

DEV1024 has now FAILED its trigger assertion. The deployed wrapper checks the
current forward's `num_tokens >= gpu_prefill_token_threshold`; the launch uses
chunked prefill 512. Total prompt length above 1024 does not establish trigger
coverage. Preserve v1 unchanged. Either exclude 1024 as incompatible with the
frozen chunk configuration in a DEV decision, or create a separately frozen
DEV experiment with appropriately sized chunks; do not call v1 a passing test.

## Unclosed warning

Both successful DEV servers emitted `storage_offset ... result is untrustworthy`
during the first layerwise migration. Exact weight assertions alone are not
sufficient to establish correct subsequent grouped MoE computation.

The standalone NZ format-29 BF16 slice-copy, NPU equality, whole-tensor ND host
oracle, and sliced host-copy probe passed, including [8,2048,1024] source shape.
The initial probe had internal format disabled and is explicitly not evidence
for NZ behavior. Corrected probes assert format 29. They did not reproduce the
warning and therefore do not clear the actual model warning.

Next required diagnosis is scoped call/operand tracing of the actual first
update and full-GPU compute, followed by a same-operand independent oracle.
Do not replace all NZ handling or modify arithmetic based only on this warning.

## Entry decision

1. Close warning attribution and any necessary minimal repair/regression.
2. Complete Dynamic clean restart and observer parity.
3. Select a compatible threshold using DEV only and commit candidate identity.
4. Run held-out and unchanged CMMLU256; do not use them for threshold fitting.
5. Timed short64/32 and medium256/64 comparison must be hook-free and isolated;
   finish/stop only owned competing work and independently audit resource logs.

Q32 stability campaign A completed 18 requests/1152 tokens. Campaign B v2 was
rejected by the existing swap admission guard; v3-resume is running with the
same guard and preserved A reference. Final restart/memory acceptance pending.
Static v6 has 12 cells collected; resource-isolation audit remains pending.

The authorized frozen model, GGUF, Q32 mask, C64 and CMMLU policy are unchanged.
No held-out, quality, or formal Dynamic performance launch occurred in this audit.
