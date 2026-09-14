# Current Call Graph and Owner Map

## All-NPU path

```text
DeepseekV2ForCausalLM
  -> DeepseekV2DecoderLayer
  -> DeepseekV2MoE router/top-k (global logical expert IDs)
  -> UnquantizedFusedMoEMethod.forward_npu
  -> Ascend init-routing-v2 / grouped-matmul / activation / finalize-routing
  -> shared expert (SGLang model layer, once)
```

## KT Hybrid path

```text
DeepseekV2MoE router/top-k (single owner)
  -> KTEPWrapperMethod.apply
     -> logical IDs split by fixed gpu_experts_mask
     -> CPU branch: KTMoEWrapper / LLAMAFILE / CPUInfer
     -> NPU branch: original SGLang UnquantizedFusedMoEMethod
     -> additive merge
  -> routed_scaling_factor at the existing DeepSeek model boundary (once)
  -> shared expert at the existing DeepSeek model boundary (once)
```

## Ownership contract

| State or operation | Owner | Invariant |
|---|---|---|
| Router logits and top-k | SGLang DeepSeek layer | Computed once |
| Canonical expert ID | Router/global logical ID | Never redefined by CPU |
| Logical-to-physical NPU mapping | KTEP wrapper metadata | CPU ID becomes `-1`; NPU IDs are dense physical IDs |
| Routing weight | Router | Applied once by the selected expert backend |
| `routed_scaling_factor` | DeepSeek model layer | Applied once after routed contribution |
| CPU expert tensors | LLAMAFILE GGUF | Same checkpoint tensors as NPU snapshot |
| NPU expert tensors | SGLang fused MoE layer | Resident BF16 weights |
| Shared expert | DeepSeek model layer | Never duplicated in KT wrapper |
| CPU/NPU merge | KTEPWrapperMethod | Sum of disjoint expert ownership sets |
| Stream handle | Active torch_npu stream | Passed as native ACL stream to CPU transfer boundary |

All-accelerator layers bypass `KTEPWrapperMethod`, preventing unnecessary CPU
buffer allocation and preserving the ordinary SGLang execution path.

