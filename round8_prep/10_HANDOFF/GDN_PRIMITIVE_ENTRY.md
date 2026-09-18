# GDN primitive entry — 2026-09-18

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
