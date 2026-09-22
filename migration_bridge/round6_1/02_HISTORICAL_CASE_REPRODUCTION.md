# Round 6.1 historical case reproduction

Evidence state: `A3_VERIFIED`.

The exact Round 6 held-out blocker was reproduced on A3 before adding any new
stagewise instrumentation. The frozen D1 placement, model revision, GGUF,
matched-history prefix, parent wheel, SGLang child, and execution profile were
unchanged.

## Reproduction result

| Field | D1_NODEFER | D1 Deferred |
| --- | ---: | ---: |
| Prompt | `hd_en_02` | `hd_en_02` |
| Token index | `43` | `43` |
| Serving top-1 | `21` | `21` |
| Anchor token 21 logit | `30.625` | `30.75` |
| Candidate token 16520 logit | `19.875` | `19.25` |
| Pairwise margin | `10.75` | `11.5` |
| Difference from NODEFER | — | **`+0.75`** |
| Exact against old capture, positions 0-43 | `YES` | `YES` |

This is the same failure recorded by the immutable Round 6 HD decision:

```text
abs((30.75 - 19.25) - (30.625 - 19.875)) = 0.75
0.75 > frozen B_pair 0.46875
```

The serving top-1 did not flip. The rejection remains valid because the V1
contract bounded pairwise distortion across the union candidate universe.

## Route interpretation

At target position 43, the Deferred run's position-aligned trace reports:

```text
target-row CPU hit count = 0
target-row deferred route count = 0
final pending set = []
```

This does not make the request a CPU-not-hit control. The matched prefix had
CPU work and a real layer-7 Deferred route, exactly as recorded by Round 6.
The target logit is therefore downstream of an earlier approximated layer
boundary.

## Artifact identity

The newly captured 44-position tensor files remain on the A3 persisted volume:

```text
/workspace/results/round6_1-repro-nodefer/capture/logits/hd_en_02.pt
/workspace/results/round6_1-repro-deferred/capture/logits/hd_en_02.pt
```

Their SHA256 values are:

```text
nodefer  c7c285e79cbf5967267387874c1d240939752e2c300ec6ee16a3109b7dc7c18a
defer    c2a7a3b1a5a28a3718483e2c63d755f828b9c8250dd6bbcfad83bc220b145c14
```

The raw tensors are intentionally not duplicated into Git. The committed
manifests, trace, command, normalized result, and checksums provide the audit
index; the A3 volume retains the large binary evidence.

## Gate

```text
WP2_HISTORICAL_CASE_REPRODUCTION = A3_VERIFIED_READY
ENVIRONMENT_DRIFT = NOT_OBSERVED
ROUND6_V1_REJECTION = UNCHANGED
```

Stagewise instrumentation and replay may now begin. D2/D3 remain forbidden.
