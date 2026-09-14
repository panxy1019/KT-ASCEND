# Current Backend Baseline

Status: `CURRENT_BACKEND_REPRO = PASS`.

The first launch from the fresh image exposed environment drift: the image did
not contain the frozen `transformers-kt` tokenizer/model package and diverged
at token 2. Copying the exact frozen `transformers-kt==5.6.0.post2` and
`sgl_kernel_npu==2026.6.1` packages from the prior disposable container restored
the frozen environment. This was an environment correction, not a model or
validation change.

With ARM LLAMAFILE SGEMM restored:

- P1: 42/45 exact
- repeated failures: `v_struct_03` at requested lengths 16/32/64
- unique first-divergence trajectory: token index 9, token 8828 vs 1273
- All-NPU margin `(8828 - 1273)`: `+0.125`
- LLAMAFILE margin: `0.0`, greedy choice 1273
- all outputs finite: yes
- prefix deterministic: yes
- post-divergence max absolute selected-token logprob delta: `2.4375150725`

The last value is trajectory-contaminated and is not used as the intrinsic
backend error measure.

Evidence:

- `evidence/round4a2-current-backend-p1.json`
- `evidence/round4a2-current-backend-compare.json`
- `evidence/round4a2-llamafile-vstruct03-top20.json`
- `evidence/round4a2-llamafile-ven01-top20.json`
