# Real Layer0/E0 LLAMAFILE512/C64 — 2026-09-18

Actual deployed package `/workspace/c64-dev/round7/backing-fix-build-v1/runtime/kt_kernel`
reports native BF16=false and AVX2 BF16=false. Hence native BF16 lane is
UNAVAILABLE_IN_CURRENT_BUILD, not a numerical failure. No ISA emulation or
unqualified backend substitution. Retain planned LLAMAFILE/F32 fallback.

Generated single-expert GGUF from the hash-verified F32 sample: gate/up split per
frozen model loader, three projections, layer0/expert0 only. GGUF SHA256:
`409ea8af33188438847c5cd6f29496e01a34f6f0762518e0c8d7c2b5d18d80d0`.
This is not a full256-expert model artifact.

Executed actual KTMoEWrapper LLAMAFILE load and forward, hidden2048/intermediate512.
Runtime logs confirm C64, two subpools NUMA4:32 and NUMA5:32, split256+256,
FP32 partials and final merge cast. No production source/binary edits.

| qlen | Relative L2 vs canonical | 10 repeats | Finite |
|---:|---:|---|---|
| 1 | 0.0016857871 | exact | true |
| 8 | 0.0016604444 | exact | true |
| 64 | 0.0016549627 | exact | true |

Status PASS_SINGLE_EXPERT_C64; every row <=0.01. Fixed synthetic operands,
seed8675309; route weight1, topk1 solely for isolated operator. This is NOT the
Qwen production top8/256-expert routing configuration. No end-to-end qualification.
GGUF loader warned about read-only NumPy views; no write was performed by the
test; preserve warning, do not declare broader loader lifetime safety from this.

Remote logs: `native-bf16-capability-v1.log`, `llamafile-sample-v1.log`; raw result
`llamafile-sample-v1/result.json`, copied locally. Process returned and CPUInfer
shutdown completed. Further gates: early/mid/late coverage,256-expert mask/top8,
GGUF identity re-read, GDN model state, isolated dependency freeze, Q64 then Q32.
No full120GiB conversion or Qwen quality run started.
