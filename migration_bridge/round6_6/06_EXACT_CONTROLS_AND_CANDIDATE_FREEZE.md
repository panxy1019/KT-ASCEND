# Exact controls and candidate freeze

Target110.129.0.22/NPU0,32CPU workers, one pool, production LLAMAFILE binary unchanged, Deferred/Dynamic/GraphOFF. ObserverOFF generation uses the same frozen model, tokenizer and four-R32 placement.

## Fresh same-path controls

First fixed prompt from each QD6 family English/Chinese/numeric/structured; no result-dependent selection. Four prompts×(10×64tokens +8/16/32prefix requests)=52requests, all exact. A wholly new serving process repeated4×64tokens, exact vs the first process. Both processes were reaped before successors started.

The observed common-history sampler check also passed4/4: first cross-backend free-generation divergence at19/14/37 for English/Chinese/numeric; structured stayed exact for64tokens. Up to and including each first divergence, observerOFF output equals the captured Hybrid natural choice on the same history. This is bounded observer parity, not proof for every untested prompt.

Evidence: `evidence/qd6_controls_v1`, `evidence/qd6_free_common_history_v1.json`. Logical prompt lengths are checked from request and response metadata; backend padded log counters are not substituted for actual request lengths.

## Candidate

`NUMERICAL_CONTRACT_CANDIDATE.json`

SHA256:`43a0bd582a4b126feb7ed851b85e1e94c1de98131700547eead4615d3e21967a`.

Generated only after QD6 audit guards and exact-control guards completed/reaped; checks actual91runtime file hashes, frozen input hashes, reviewed QD report hashes, and refuses if the HD6 output root already exists. Target file was frozen before launching HD6, and a byte-identical copy was committed locally. Candidate is not a final contract approval and does not authorize quality before held-out review.

HD6 capture now runs once, All-NPU then identical-history four-R32. No refit. After capture, the bounded driver runs transfer/logit/router audits and awaits review; it never silently starts quality or all-MoE promotion.

## Reference runtime supplement

`evidence/reference_runtime_v2.json` records the CPU reference's PyTorch2.9.0+cpu build (GCC13.3, BLAS_INFO=open), bundled OpenBLAS, OpenMP and ARM Compute libraries, and resolved linked-library hashes. This supplements provenance after B0; it does not alter the original B0 result or substitute a new production backend. Installed CPU extension remains the original LLAMAFILE artifact.
