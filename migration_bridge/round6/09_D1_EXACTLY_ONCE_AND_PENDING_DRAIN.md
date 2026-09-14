# Round 6 D1 Exactly-Once and Pending-Drain Evidence

Evidence state: `A3_VERIFIED` for the specific mechanism gates below; full
D1 generation, numerical, quality and lifecycle gates are separate.

## Real-weight incremental reconstruction

The isolated LLAMAFILE harness used the frozen Round 6 D3 F32 GGUF
(`4b42799ec7f143569692133aadd37f089ff0c18508411e5fd37911b62c8e323f`)
and the frozen D1 Layer 7/8 CPU expert sets. It compared full Layer 7 output
against immediate + deferred, and Layer 8 incremental output against its own
work + the previous-layer deferred work.

```text
Layer 7 split reconstruction: byte exact, max_abs 0
Layer 8 incremental reconstruction: byte exact, max_abs 0
Recovered deferred partial from merged output:
  max_abs 5.960464477539063e-08
  relative_l2 2.5475927821528976e-08
100 repeated executions: one unique output hash
```

The small nonzero error in the reverse subtraction is ordinary FP32
subtraction rounding; the forward split and incremental reconstructions are
byte exact. Harness payload `pass=true`, canonical payload SHA256
`bbf9ed585f5fdd40516b3caa192542f5f261fdca23584e3f800324c845941eb8`.
Raw result JSON SHA256:
`200c6521075285d2733780ed5252fd374a3451dc8a0fa11dd40b9532809ba595`.

## Final production-path trace

With the final pending-state consumption code installed, four S6 prompts
(English, Chinese, math, structured JSON) generated 32 tokens each under
production overlap. The trace was default-off instrumentation enabled only for
this probe. The analyzer used `--producer-consumer 7:8
--require-final-drained`.

| Layer | Submit / sync | Immediate CPU routes | Deferred CPU routes | Pending | Consumed incremental | False / missed pending |
|---:|---:|---:|---:|---:|---:|---:|
| 7 | 134 / 134 | 162 | 13 | 13 | 0 | 0 / 0 |
| 8 | 134 / 134 | 137 | 0 | 0 | 13 | 0 / 0 |

The selected Layer 7 CPU experts `{16,38,41,50}` and Layer 8 experts
`{8,25,33,55}` were all observed. The producer/consumer count was exact
(`13 == 13`), and the final pending state was `[]`. The analyzer reported no
failures over 804 trace records. Trace SHA256:
`8406333210e91fa2b2a500e0e2c25bf9e7750ddf7591ccd866ab032a4626b8ba`.
The analyzer payload SHA256 is
`efa0eb24f4275e898757c9907ae1e4cfa55a251c0ad865d0043798c3f8186d8e`.

Raw A3 artifacts and SHA256 manifests:

```text
/home/admin/kt-artifacts/round6-deferred/d1-incremental-final/
/home/admin/kt-artifacts/round6-deferred/d1-trace-final/
```

## Gate

```text
D1_REAL_WEIGHT_SPLIT_AND_INCREMENTAL = A3_VERIFIED
D1_100_REPEAT_OUTPUT_DETERMINISM = A3_VERIFIED
D1_PRODUCTION_PENDING_CONSUMER_EXACTLY_ONCE = A3_VERIFIED
D1_TERMINAL_PENDING_DRAIN = A3_VERIFIED
D1_FULL_MODEL_QUALIFICATION = IN_PROGRESS
```
