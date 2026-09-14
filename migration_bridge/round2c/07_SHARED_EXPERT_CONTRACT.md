# Shared Expert Contract

```text
SHARED_EXPERT_OWNER = outer model layer
SHARED_EXPERT = BYPASSED_WITH_CONTRACT
```

## Evidence

Current SGLang `DeepseekV2MoE` constructs routed experts separately from optional shared experts (`third_party/sglang/python/sglang/srt/models/deepseek_v2.py:493-509` and `:552-575`). In the normal forward path it computes shared output at lines 761-767, routed output at 806-816, and adds shared output once at 818-830. `_forward_shared_experts` is defined at lines 1299-1307.

## Round 2C boundary

The new coordinator returns only:

```text
y_routed = y_cpu_routed + y_npu_routed
```

The future model integration owner must produce:

```text
y = y_routed + y_shared
```

exactly once. The coordinator must not load, execute, fuse, or add shared experts while the outer model retains ownership. This avoids duplicate contribution and avoids premature SGLang/model changes. No full model was loaded in Round 2C.
