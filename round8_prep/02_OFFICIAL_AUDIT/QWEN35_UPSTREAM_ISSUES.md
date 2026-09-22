# Issue / PR registry

Checked 2026-09-18. Closed issue ≠ fix present in our checkout. No patch applied during prep.

| Source | Evidence and applicability | Required verification |
|---|---|---|
| [PR22674](https://github.com/sgl-project/sglang/pull/22674) | Page shows Closed, not sufficient proof of merged ancestry. Proposes GDN packed quantization mapping qkvz→qkv/z, ba→b/a | Identify actual quant loader and selected GDN representation; BF16 path may not use quant scheme lookup |
| [Issue23656](https://github.com/sgl-project/sglang/issues/23656) | Qwen3.5 NPU missing interpolation argument; reporter contrasts transformers5.3.0 vs5.5.4/5.6.2. Multimodal/TP4 repro | Text-only processor/config import smoke; do not downgrade global environment from one issue |
| [Issue34003](https://github.com/sgl-project/sglang/issues/34003) | Dynamic CPP/radix NPU long context repro, PP4/TP4 | Initial TP1 GraphOFF/radixOFF scope avoids that combination, not a proof issue fixed |
| [Issue23652](https://github.com/sgl-project/sglang/issues/23652) | NPU Qwen3.5 prefix-cache report | RadixOFF initial gate; reconsider only in separate cache qualification |

Missing source details remain explicitly open in patch queue. No issue provides permission to loosen quality, change model, or skip real operator tests.
