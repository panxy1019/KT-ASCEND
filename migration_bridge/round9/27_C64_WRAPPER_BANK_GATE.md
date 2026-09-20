# Round9 real BF16 bank and C64 wrapper gate

2026-09-20, 110.123.0.3, isolated logical0 -> physical15. Scope is CPU expert
partial calculation plus framework NPU transfer. Full KTEP/NPU expert merge,
full-model admission and full-model tests remain unqualified.

## Identity and implementation

- Original checkpoint revision: de4b8e4d43b917e7706784d8bb445c9af86a3540.
- Layer0 GGUF: 5,033,165,120 bytes; SHA256
  731679cb66e83c7c9a2b4b2ae29af68581729e194d8d495530f2aec1342bd97b.
- 512 experts x gate/up/down =1536 projections verified bit-for-bit against
  original safetensors, including axes. Storage remains BF16; no F32 bank.
- The Python wrapper now permits C64's 320+320 intermediate partitions only
  when the compiled native extension advertises the BF16/F32 tail capability.
  Older extensions fail closed. Mixed storage and blocked quantized formats
  retain their previous alignment restrictions. Configuration tests4PASS.
- Fresh Ascend-enabled binary passed framework callback/stream/transfer tests
  (5 tests, including1000 round trips for BF16/F32/int64). A subsequent route
  indexing fix rebuilt the same binary and exercised its transfer path in the
  real wrapper test. Each probe records its own binary SHA; do not conflate them.

## Failures and repairs

1. v1 aborted before loading: NUMA memory binding failed. Minimal libnuma
   set_mempolicy returned-1/errno1 with default container capabilities and0/0
   with SYS_NICE. v2/v3 add only SYS_NICE, matching the earlier C64 setup.
2. v2 mixed routing passed (relativeL2 2.57e-7); resident-only CPU output was
   exactly zero. Repeated CPU expert511 failed at16.0873% relative error.
   `forward_one` looked up route weight by expert ID and always selected the
   first occurrence. The repair saves each active contribution's original
   top-k position and uses that weight. No threshold relaxation.
3. v3 after rebuild passed all9 cases. Failed containers/logs/JSON are retained.

## Verification and bounds

Actual `LlamafileMoEWrapper.load_weights` loads the entire original512-expert
bank. Two NUMA pools6/7 have32 workers each. Container CPU quota72, allowed
CPUs480–639 (whole nodes required by native topology validation), RAM/swap20GiB,
networknone, pids512, only davinci15 and driver control devices. These are
functional shared-host checks, not an exclusive-core performance qualification.

Resident owner mask is logical_id%8==0 (64 owners). T1/10/32 each exercise
mixed CPU/resident routing, resident-only routing, and ten repeated CPU routes.
Each case repeats3 times with identical hashes. Maximum relativeL2 is
0.0001138027073466219; all outputs finite. Resident-only CPU partial is exactly
zero. Cleanup reports allocated=0 and reserved=0. The oracle independently
uses original safetensor projections, BF16-visible intermediates/routing
weights and FP32 down partial accumulation.

Evidence: `evidence/physical15/20260920-wrapper-bank/` includes red/green logs,
container configs, result JSON, native binary digest, tested-source hashes and
SHA256SUMS. Bank manifest is in `evidence/physical15/20260920-bank/`.
No qualification for the other47 banks is implied; their CPU-only conversion
and complete identity checks run separately and must finish successfully.
