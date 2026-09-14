# P1 Controlled Requalification

OpenBLAS passed the controlled integration gate before full P1:

- C0–C4 explicit `ΣCPU + ΣNPU` comparison: max error 0
- shared expert and routed scaling exactly once: max error 0
- CPU-not-hit vs All-NPU: max error 0
- sequential routed hash: `1498de1aaccc7a387aaf63417d28fe3ba812422828a0c6598c86565f5bdb8875`
- overlap routed hash: identical
- final hash: `8c1159ebf0e7e3ddca84f99a82318e9060d9a0f599b9d0b71444d1acaa9a7d8b`
- 1000-forward unique routed hashes: one
- registered SGLang Ascend KT EP routing test: 1 passed

This proves the full-model failure is numerical behavior, not integration,
ownership, routing, overlap, or lifecycle nondeterminism.
