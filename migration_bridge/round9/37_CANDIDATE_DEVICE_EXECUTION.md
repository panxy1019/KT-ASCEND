# Round9 candidate device execution — 2026-09-20

Latest user authorization: fresh CPU/NUMA/shared-HBM admission → C128 real experts;
low-bit and Dynamic integration gates → full-model combinations. No performance,
no schedules, no push/main merge/tag. Host110.123.0.3 physical15 remains shared.

## Execution order and acceptance

1. Fresh physical-core/NUMA/HBM admission, bounded C128 BF16 wrapper.
2. Real C128 KTEP merge with64 original NPU experts; exact processed-weight identity and ownership.
3. Dynamic device promotion/demotion and native mask/map consistency, then actual
   scheduler request cleanup integration; no full-model Dynamic before both pass.
4. Low-bit320/160 native decode/grouped-prefill kernels, exact converter/loader
   identity and quantized-reference numerics; no dtype/partition guard bypass.
5. Separately admitted candidate full-model load/compact/restart/cancel/resource;
   complete frozen baseline256 before candidate quality comparisons.

## Resource ruling

At22:09UTC physical15 had no NPU processes; board2880MiB, training container still
running. Preserve user15GiB allowance PLUS current driver baseline,8GiB reserve.
Do not claim concurrently active training validation from this idle observation.
New unknown NPU process stops only our probe; no training process is manipulated.
C128 uses NUMA2,3,6,7, four32-worker pools and640/4=160 partitions.
Only even logical CPUs in these nodes:160 distinct physical cores allowed,
CPU quota136=128+8aux.64GiB RAM cap/no swap for bounded layer0 tests,4GiB owned
HBM guard, framework allocator fraction.04, timeout600s. Not full-model admission.

## Results

- v1 wrapper PASS9cases×3repeats, maxrelL2 .00011380300566088408 <.01.
  Actual128 pinned worker threads,32 per selected NUMA, no SMT duplicates.
  Source receipt144 live sizes + metadata checked; layer0 bank full SHA verified.
  Exit0/OOMfalse, framework allocated/reserved both0 after cleanup.
  Observed board2943MiB/own118MiB (sampled, not absolute peak guarantee).
- v1 guard encountered docker-top exit race after natural process completion;
  conservative own-stop issued. Retain raw guard error, do not call guard clean.
- v2 real KTEP local merge PASS, exit0. Device/module qualification only.
- v3 Dynamic device promotion/demotion/promotion PASS, all64 resident weight
  identities checked after each transaction, pinned native mask pointer stable,
  post-migration KTEP numeric/ownership/repeat gate PASS. Exit0.
- Native low-bit tile v1/v2/v3 FAILED with nonfinite Q8_0 output, including tightly
  packed inputs. Root cause: ARM IQK Q80 expects block_q8_0_x4 activation groups;
  caller/from_float supplies standard Q8_0. Fix dispatch Q8_0 weights to existing
  tinyBLAS standard-layout path. Isolated v4 native24 shapes PASS,including
  padded lda/ldb/ldc,160/320 reduction,2560 gate input,64 tail,1/10/32 tokens.
- Isolated candidate native build PASS; baseline native library not overwritten.
  Low-bit multi-pool capability export added,32-aligned Q8_0/MXFP4 TP and
  forward_many tail path gated on this export/ARM DOTPROD/noSVE.
- Actual original10-expert CPU MoE: Q8_0 and MXFP4 × C64 and C128 ALLPASS,
  each6 token lengths(1,2,9,10,11,32)×3 repeats, ownership flip exact.
  Reference explicitly includes quantized weights AND Q8_0 activations.
  This is kernel correctness, not accuracy relative to original BF16 model.
- Vector encoder1025blocks byte-identical to scalar codec; full tiny GGUF output
  SHA identical for both formats.3 ARM tests PASS. Original scalar remains default.
- Layer0 full512 expert conversion for both formats RUNNING in
  kt-round9-lowbit-bank-v1 (4CPU8GiB,noNPU,1200s per format).
- Guard normal-exit/docker-top race repaired;10 targeted local testsPASS.
  Updated guard has not yet run in another real candidate container.

Low-bit inspection: repository patch0001 adds actual MXFP4 vec-dot support to
pinned ggml; unpatched /tmp llama is not the built dependency. Need audit actual
llamafile sgemm wiring and all decoder/prefill calls, not assume enum support
implies grouped GEMM support. Preserve baseline native binary and SGLang tree.

## Actual layout correction —22:33UTC

v6 explicitly queried torch_npu.get_npu_format: bothw13/w2 are ND(2). Earlier
v2–v5 test labels saying NZ were inaccurate: format_cast returned base format
with this runtime configuration. Preserve raw logs, correct interpretation:
weight bytes and migration/merge correctness PASS for actual processed tensors;
NO NZ qualification. New adapter rejects NPU formats other than ND2. New probe
records actual formats and uses processed-weight labels. v6 candidate-binary
BF16+C128+Dynamic local merge PASS, no change to baseline binary.

Installed real f02 normal event-loop method executed with CPU-only scheduler
doubles: chunk completion and abort both prove pre-forward lease and post-release
commit hook ordering. PASS_ACTUAL_F02_LOOP_HOOK_ORDER_ONLY; this does not replace
fullmodel lifecycle/cancel testing. Fullmodel resource launch awaits user's
clarification whether15GiB covers training only or all non-Round9 board use.
Q8 full48 CPU conversion currently RUNNING kt-round9-q8-all48-v1.

## Review corrections and exclusive full-model transition —22:48UTC

User explicitly changed physical15 from shared to exclusive until further notice.
Fresh npu-smi shows no process and2880MiB board baseline. Existing training
container remains untouched. Timers remain paused. No performance work.

Native review found low-bit fp32_tp_partial_ omitted Q8_0/MXFP4. Existing wide
MoE reference tests had hidden premature BF16 rounding before TP merge. Old
binary strict C64/C128 identical-input regression fails at q1, relativeL2
0.0021650733. Corrected isolated buildv2 passes bothformats×C64/C128, sixq lengths
and3 repeats; cross-pool maxima Q8_0=1.24856e-7, MXFP4=1.06529e-7 (gate1e-5).
Quantized weight+activation reference maxima Q8_0=.000256949, MXFP4=.000566466.
The initial new regression launch missed SYS_NICE and exited134 with NUMA binding
failure; corrected CPU-only launch retained asred-v2/green-v1. No NPU touched.
Earlier low-bit MoE and KTEP results are superseded by corrected-native runs.

New real C128 full512 bank+64resident KTEP+3 Dynamic migration device gates:
v7Q8_0, v8MXFP4, v9BF16 all PASS_REAL_KTEP_LOCAL_MERGE with actual ND2 enforced.
Each launch binds frozen probe source, actual native/Python runtime and preserved
SGLang Python tree SHA. These remain local-layer gates, not full-model claims.

Independent review repaired candidate isolate fallback without profile, Dynamic
env-only bypass, and candidate storage not bound to actual GGUF bank. Candidate
model now requires explicit profile; all48 real layers bind the same profile and
verified source/bank identity before Dynamic can bind. CPU verifier rehashes all
48 complete bank files and checks actual types/shapes/file set. Profile, catalog,
source receipt and unchanged read-only live bank signatures are enforced at load.
Installed SGLang-v3 tree receipt binds both preserved baseline and candidate tree;
launcher and server recheck actual imports against payload/admission hashes.
Extra source snapshot recheck added for subsequent launches; v10 snapshot was
already frozen and unchanged when launched.

Full48 Q8 conversion complete; MXFP4 CPU conversion still running. Original
conversion catalog validates receipts/sizes only; newly added full-byte/type
verifier is mandatory before model use. Converter code for future runs now uses
explicit exceptions (not Python assert) and rehashes banks before publication.
Do not replace a running converter's source or reinterpret its old catalog as
full-byte verification. BF16 full48 verifier passed; Q8 verifier running.

29 targeted ARM CPU tests PASS.38 local CPU tests PASS. A broader local discovery
attempt included numpy-dependent test_candidate_gguf and failed import because
host lacks numpy; the same test passed inside ARM image. This is not a hidden
failed device result. Independent follow-up review found no remaining P1.

C128/BF16/static fullmodelv10 STARTED under exclusive fresh admission,128 workers,
NUMA2/3/6/7,136CPU quota,640GiB RAM/no swap,41GiB modelHBM/8GiB board reserve.
First HTTP, system8 repeat checks, canaries and4096resource requests queued in
that owned container. Results remain PENDING until actual completion receipts.
