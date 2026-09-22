# Static HOLDOUT routing: prefill and decode separation

Date: 2026-09-17. Evidence: **ANALYTIC_DERIVED_FROM_CLOSED_A3_CAPTURES**.
This is a lossless offline analysis of previously closed A3 captures, not a new
A3 execution, performance qualification, or scheduler selection.

## Source and integrity

- Frozen HOLDOUT128 SHA256: `1e1c8287191007ec6ae3a91ed974d3648b7c1b33d46170a60638c54eedc1dbf5`.
- Uniform original archive: `raw/uniform-holdout-evidence-v1.tar.gz`, SHA256
  `835a2dcc59d8fc93b997cce49d6a62905f4e83fb67616c09819a24215deca78e`.
- Frequency original archive: `raw/frequency-holdout-evidence-v1.tar.gz`, SHA256
  `9e7dddc576c8cf409c41324015dc47d9b4a275dc5ca98c270cb11e7140b1a1ef`.
- Machine output: `stage-routing-v2/static-stage-routing.json`, SHA256
  `3267145ac84a0a00176725ed5f16d356b30670f692c6f3599c1b1db06506798b`.

Both lanes have 128 requests with 16 returned tokens. Exact prompt IDs,
input lengths, returned token histories, positions and recording epochs
re-associate 128 prefill forwards and 1,920 useful decode forwards. The 128
proven unused final-token forwards are excluded; unknown passes=0. Each included
forward has all 26 routed layers. Prefill plus decode assignments, hits and
weight sums agree with the previously frozen combined routing summaries.

The first local parser attempt assumed both archives had a `runtime-v2/` prefix
and failed before producing an output. Uniform actually uses a flat root and
Frequency the nested root. The repaired parser explicitly accepts exactly one
of these layouts, rejects ambiguous roots and writes a new output directory.
Original archives, captures and qualification results were not overwritten.

## Stage comparison

| Metric | Uniform prefill | Frequency prefill | Uniform decode | Frequency decode |
|---|---:|---:|---:|---:|
| Model-token input rows | 126,112 | 126,112 | 1,920 | 1,920 |
| Routed assignments | 19,673,472 | 19,673,472 | 299,520 | 299,520 |
| NPU-hit assignments | 2,448,460 | 5,212,963 | 36,290 | 32,978 |
| CPU-hit assignments | 17,225,012 | 14,460,509 | 263,230 | 266,542 |
| NPU-hit rate | 12.4455% | 26.4974% | 12.1161% | 11.0103% |
| CPU route rate | 87.5545% | 73.5026% | 87.8839% | 88.9897% |
| CPU route-weight mass | 87.0194% | 73.9852% | 88.3079% | 90.2918% |

Frequency reduces prefill CPU assignments by 16.0494% relative and CPU
weight-mass fraction by 14.9785% relative. However, decode CPU assignments
**increase** by 1.2582% and CPU weight-mass fraction by 2.2465%; decode NPU-hit
rate is 1.1058pp lower. The earlier combined improvement is prefill-dominated.

This finding does not reject or select Frequency by itself. Frequency quality
remains PASS173/256 under the frozen observed -5pp policy. Static selection
awaits all 36 observer-OFF ABBA cells and independent host-window validation.
Later Dynamic comparisons must report measured decode-only exposure separately
from full-device prefill, and distinguish actual Static runs from initial-mask
counterfactuals on Dynamic routes.

## Reproduction

```bash
python3 migration_bridge/round7/tools/analyze_static_decode.py \
  --uniform-archive migration_bridge/round7/raw/uniform-holdout-evidence-v1.tar.gz \
  --frequency-archive migration_bridge/round7/raw/frequency-holdout-evidence-v1.tar.gz \
  --corpus migration_bridge/round7/01_STATIC_FREQUENCY/corpora/SCHED_HOLDOUT.jsonl \
  --output <new-nonexistent-output-directory>
```

No NPU execution, model arithmetic change, corpus refitting or mask update is
required. Counterfactual Dynamic fields are deliberately absent from this Static
report. Historical numerical rejections remain unchanged.
