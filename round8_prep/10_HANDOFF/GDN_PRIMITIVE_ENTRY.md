# GDN primitive entry — 2026-09-18

20:59 update: primitive v1 completed PRIMITIVE_SMOKE_PASS. Decode output
[1,1,32,128], prefill output[1,64,32,128], both finite and3-repeat exact.
Raw JSON copied locally. Launched separate `gdn_state_probe.py` four consecutive
decode steps with retained state versus explicit CPU recurrence. Result/log
`gdn-state-v1.json`/`.log`; diagnosis only, no model promotion. Check active
PTY10100 before issuing shell commands or restarting.

Continuous-state probe also completed: all4 outputs/state finite; output
relative-L2 versus CPU recurrence was0.00164455/0.00166164/0.00165656/0.00167085.
State norms31.8870/35.4269/37.3120/36.6112 prove nonzero state persisted/changed.
This is descriptive numerical evidence, not a pre-registered model/cache gate.
Next: causal convolution, actual SGLang state integration and isolated build
provenance, plus256-expert/top8 integration; do not rerun completed probes.

Actual deployed `linear/kernels/gdn_triton.py` replaces generic kernels on NPU:
`chunk_gated_delta_rule_npu` for prefill and
`fused_sigmoid_gating_delta_rule_update_npu` for decode. The Triton class name
alone is not evidence of a CUDA-only path. Actual package function source saved
remotely at `/home/admin/kt-round8-qwen35/gdn-api-audit-v1.log`.

Launched `tools/gdn_primitive_probe.py` in the isolated Round8 container:
frozen config dimensions16 key heads,32 value heads,K=V=128; seed8675309,
decode1 and prefill64, each3 repeats with reset zero state. Synthetic BF16 inputs,
FP32 state, q/k normalization and per-sequence offsets. No model weights/cache
allocation or serving process. TRITON_CACHE_DIR=/workspace/round8/triton-cache,
so generated kernels do not enter the shared Round7 mounted paths.

Remote result/log `gdn-primitive-v1.json` / `gdn-primitive-v1.log`.
At launch no PASS is claimed. Inspect active process/output before rerun; initial
compilation can take time. Owned root PTY10100 currently runs the foreground
docker exec, do not send new shell commands until it completes.

Even a primitive finite/repeat PASS is not a recurrent-state numerical oracle,
model construction, causal-convolution or end-to-end GDN cache qualification.
Q64/Q32 remain NOT_RUN. Do not inherit DSV2 model correctness.
