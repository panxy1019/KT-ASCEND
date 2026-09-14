# Round 4A1 Base and Blocker

## Frozen source

- parent base: `08c84e3d25b7ed473dd9efad7768485c1392cb30`
- parent branch: `feature/kt-round4a1-numerical-closure`
- SGLang child base: `06b319dc3a62b77c880e36b042d273bfc3957d12`
- child branch: `feature/kt-ep-round4a1-numerical-closure`
- all-NPU 45-case baseline SHA256:
  `19db4e413d5548bdd8c0467915b7daf093057ab64eec5279745c123489bf042e`
- original P1 45-case SHA256:
  `ba1a77ea796f512003532ccb7f9b645b5193be2d7d88cbf31586068785cb276c`

The A3 work used only the disposable `kt-r3-dsv2lite` container, CPU cores
0-15 and NPU0. No host dependency, system package, driver, toolkit, service or
business container was changed.

## Frozen P1 contract

- model: `DeepSeek-V2-Lite-604d5664`
- TP: 1
- CPU placement: Layer 17 experts `{6,8,25,36}`
- accelerator experts at Layer 17: 60
- CPU method: LLAMAFILE, F32 GGUF
- validation: 9 prompts x token counts `{1,8,16,32,64}` = 45 requests
- hard gate: 45/45 exact token IDs and maximum absolute logprob delta <= 0.20

## Entry blocker

The frozen Round 4A P1 run completed and was deterministic, but only 30/45
requests had exact token IDs. Maximum absolute logprob delta was
`2.4378519617021084` after divergent history. P2 and P3 were therefore not
eligible to run.

## Current gate state

After the two verified minimal numerical fixes, P1 is 42/45 exact. All three
remaining mismatches are the 16/32/64 prefixes of `v_struct_03`; they share one
first divergence at generated token index 9. The post-divergence maximum
absolute logprob delta is `2.4375150725245476`, so P1 remains blocked and P2/P3
remain `NOT RUN`.
