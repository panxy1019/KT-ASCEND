# P1 Full-Model Requalification

Status: `FAIL — STOP BEFORE P2`.

Frozen matrix: 9 prompts x 1/8/16/32/64 = 45 requests.

- exact token requests: 42/45
- mismatches: `v_en_01` at 16/32/64 tokens
- unique first divergence: token index 10, All-NPU token 30 vs OpenBLAS token 279
- prefix deterministic: yes
- finite: yes
- post-divergence max absolute selected-token logprob delta: `2.0071879169`
- matched-history critical margin `(30 - 279)`: All-NPU `0.0`; LLAMAFILE `0.0`; OpenBLAS `-0.125`
- CPU placement and corpus: unchanged

The post-divergence maximum exceeds 0.20 but is trajectory-contaminated. The
matched-history result independently fails the exact-token gate, so no
acceptance ambiguity exists.

Evidence: `evidence/round4a2-openblas-p1.json` and
`evidence/round4a2-openblas-p1-compare.json`.
