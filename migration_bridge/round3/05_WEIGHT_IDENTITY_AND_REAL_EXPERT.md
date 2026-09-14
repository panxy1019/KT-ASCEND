# Weight Identity and Real Expert Verification

## Artifact provenance

- Source snapshot: DeepSeek-V2-Lite revision `604d5664`
- CPU GGUF: `/workspace/kt-src/artifacts/round3/deepseek-v2-lite-layer17-f32.gguf`
- GGUF SHA256: `a16a50827ec81b54195bf246c7f9d05f7c1d5f3601ee33426c732f65892e180f`
- Export: layer 17, all 64 routed experts, F32 gate/up/down tensors
- DeepSeek routed intermediate size: 1408; LLAMAFILE TP1 tail-safe path used

## Final real expert

- Global logical ID: layer 17, expert 8
- CPU dtype: F32 GGUF compute/output rounded to BF16 for comparison
- NPU dtype: BF16
- Placement SHA256: `05bae81924d79677c2ea03cbce4b74b6fa6e95e144389c6cdd4890fc4ad30f53`
- CPU repeat count: 10
- CPU byte determinism: exact; maximum repeat difference 0
- CPU versus BF16-rounded FP32 reference relative L2: `5.29e-5`
- CPU versus NPU relative L2: `0.004299`
- All outputs finite: yes

The comparison used tensors from the same checkpoint revision and the same global
expert identity; it did not compare unrelated converted checkpoints.

