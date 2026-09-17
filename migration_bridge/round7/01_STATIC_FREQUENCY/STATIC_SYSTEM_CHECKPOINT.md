# Round7A static system checkpoint

Evidence state: **A3_VERIFIED**. This is not Round7 completion or Frequency
product qualification. Native arithmetic and serving source are unchanged.

## Preserved evidence

Lossless selected-evidence archive:
`../raw/static-system-evidence-v1.tar.gz`

SHA256:
`f977731ff877dcaf2fdd50ecb1aa7aa676382a56b489b62dace5248ca7c55045`

The archive preserves Uniform compact/restart requests, structure and routing
captures, process logs, resolved commands, preflight hashes and lifecycle/reaping
records, plus all six operator result JSONs and their summary. Full operator
tensor/native partial/weight dumps remain on A3 under
`/workspace/c64-dev/round7/runtime-v2/operator-representatives/`; they were not
copied into this archive. Input/output hashes are in the preserved result rows.

## Operator representatives

Reuse of the frozen actual C64 native extension and same-operand oracle:

| Layer | CPU workers/subpools | Cases | Maximum recorded relative-L2 |
|---|---|---:|---:|
| 1 | 32 / 1 | 22 | 0.0009405782341776842 |
| 1 | 64 / 2 | 22 | 0.001673673779008822 |
| 13 | 32 / 1 | 6 | 0.0012146409043853155 |
| 13 | 64 / 2 | 6 | 0.0017029279396063187 |
| 26 | 32 / 1 | 22 | 0.0011183248339594006 |
| 26 | 64 / 2 | 22 | 0.0016933269530168054 |

All 100 case rows pass the unchanged local 0.01 safety gate, ten-repeat
CPU exactness and CPU-not-hit zero control. C64 cases also pass native FP32
partial merge exactness. The NPU serving profile remains TP1: two CPU subpools
are not NPU TP2.

Experts 1 and 63 use three real historical decode operands per layer. Layers
1/26 additionally exercise qlen2/8/9/10/11/16/32/64 by stacking real decode
rows. These are operator shape-boundary diagnostics, **not captured contiguous
prefill sequences** and not exhaustive expert certification.

## Uniform compact/restart

- 32 compact requests, four independent clean-restart requests.
- Same-path trajectories and shorter prefixes exact.
- Restart and observer ON/OFF parity exact.
- Finite output, all-layer mask/mapping/ownership and BF16 physical layout pass.
- Structure SHA256:
  `c35c227f9f61fe157e8c28247c85e582bf469ee30d8f369b2557fbc4a8ca06ca`.
- Both serving process chains have completed owned reaping.

These synthetic scheduler prompts are not the CMMLU quality benchmark.

## Remaining gates

Frequency compact/restart is running normally in release-v6. The healthy
static driver will continue to independently frozen HOLDOUT routing, Frequency
CMMLU and observer-OFF exploratory performance. No Dynamic or Deferred result
is established by this checkpoint. Do not change the running release or restart
the healthy model process.
