# Patch queue — not applied

| Priority | Gap | Minimal action | Validation |
|---|---|---|---|
| P0 | Final Round7 production base unavailable | Read final state and exact runtime overlays after closure | Provenance match, no dirty-tree copy |
| P0 | ARM CPU BF16 unverified / F32 packed loader unverified | Probe native BF16; otherwise packed Qwen→GGUF adapter | Expert identity, three-way relL2≤0.01, repeat exact |
| P0 | GDN/NPU dependencies differ across snapshots | Locate actual NPU operator/quant loader; compare separate vs fused input projections | Config/model construction, GDN prefill/decode |
| P0 | Shared/routed expert integration not executed | Reuse KTEP, adapt only model boundary if dispatch missing | Route/scaling/shared once, physical mapping, finite |
| P1 | Transformers/processor compatibility | Isolated dependency resolution, no global downgrade | Text-only config/tokenizer/template and model load |
| P1 | LLAMAFILE512/C64 topology | Preserve proven lifecycle, dimension-specific validation | Decode+prefill operator and memory tests |

Each actual patch needs source pin, symptom, minimal repro, before/after and regression. No speculative production fixes during prep. No wholesale upstream upgrade; main SGLang serves comparison only.
