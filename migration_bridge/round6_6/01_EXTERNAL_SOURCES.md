# External numerical precedents

Evidence state: SOURCE_INSPECTED, not KT-ASCEND hardware verification.

* [Fiddler authors' repository](https://github.com/efeslab/fiddler) and [paper](https://arxiv.org/abs/2402.07033): CPU compute reduces expert-weight movement; directly relevant execution split, not a numerical acceptance threshold source. Paper metadata also checked with `hf papers search`.
* [MoE-Infinity DSV4 README](https://github.com/EfficientMoE/MoE-Infinity/blob/4b94d3d6e1762f8831099746aaf1212a1272a816/moe_infinity/models/deepseek_v4/README.md): native dequant/BF16 expert reference and TileLang activation-rounding divergence are documented. IMPORTANT: the README explicitly labels its historical measurements as missing exact original commit, stack, and capture-date provenance. Treat this as a design precedent, not independently reproducible proof of our backend or an imported bound. Inspected repository HEAD: `4b94d3d6e1762f8831099746aaf1212a1272a816`.
* [Mixtral-Offloading authors' repository](https://github.com/dvmazur/mixtral-offloading): quantized expert caching/accelerator computation is a different design from CPU execution. It does not resolve our same-operand CPU/NPU discrepancy.
* [Klotski paper](https://arxiv.org/abs/2502.06888): expert-aware multi-batch pipelining addresses I/O/computation overlap. Relevant to later scheduling, not evidence that this static numerical failure passes.

Adopt only the independently testable reference-backend seam. No imported speedup, threshold, CPU/NPU bitwise claim, or numerical-aware placement policy.
