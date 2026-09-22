# QD2 input freeze

128 authored prompts: English24, Chinese24, code32, JSON24, math24. Per prompt:
64 frozen-baseline-history positions. No inference or route-based selection was
used in constructing these prompts. Input IDs were generated offline on A3
with the existing pinned tokenizer; no model was loaded.

Manifest: `corpora/frozen-corpora/qd2.json`.
Canonical payload SHA (excluding its sha256 field):
`ca8b8ba8e15585c93467cd2f26a80939a2de69775ab562bc5f91cb54963081a9`.
Serialized file SHA256:
`57a807a8ea24f84137434c929e4f6546eeffd01278997e9272d3afd2844194e1`.

HD2 was generated in the same freeze invocation before any QD2 output. Normalized
text duplicates across both sets and historical corpus files were rejected.
The manifest records every exclusion file hash. This is exact-text checking,
not proof that two topics cannot be statistically dependent.

Distinct authored tasks are the intended prompt blocks. Randomized order is
for deterministic domain mixing only; it does not establish IID sampling.
The predeclared between-prompt exchangeability assumption limits statistical
interpretation to this workload design, not arbitrary production requests.
The input corpus is not the separate frozen downstream-quality benchmark.
Actual exposure remains UNKNOWN until execution; do not replace prompts if low.
