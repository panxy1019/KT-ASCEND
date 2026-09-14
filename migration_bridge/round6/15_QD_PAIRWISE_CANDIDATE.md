# Round 6 D1 QD pairwise candidate freeze

State: `A3_VERIFIED` for the frozen QD measurements and `ANALYTIC_DERIVED`
for the **candidate**, not `HELDOUT_VERIFIED`. HD model outputs were not read
or used to choose these bounds. D1 cannot be promoted until independent HD,
CPU-not-hit, quality, lifecycle, and retained gates close.

## Matched-history provenance

- D1 placement, model, GGUF, NPU0, batch 1, greedy and production overlap were
  unchanged between runs. The sole treatment was Deferred budget 0 versus 1.
- Both sweeps used `evidence/qd/defer0_history.json`, frozen before either
  full-logit sweep. Corpus SHA:
  `fe28d7fe4b4421b53901950b41a7d98b58650de359bd5141e9d05ef71e6bf6f1`.
- Both complete manifests contain 16 prompts × 64 positions = 1024 rows,
  identical prompt/token indices, history SHA, and position-aligned route
  telemetry. All 1024 requests drained pending state.
- In defer=1, 430 positions had a CPU route in the target row and 79 positions
  had an actually deferred target-row route. Large debug tensors and raw traces
  remain in the A3 disposable artifact directory; their per-prompt logit
  hashes are committed under `evidence/qd/`.
- Comparison code was the locally committed tool with SHA256
  `d4faeb8bd793ee651b4f90af6bebb6089bb5f385ec3a33d877303b5acb5d8c2f`.
  It was copied into the A3 disposable results directory without modifying
  the A3 source checkout. The running wheel/source production commit remained
  `e3084d309d6f39b8a9f7185d36ac27e4b77526d7`.

## QD distribution

Candidate universe: union of defer=0 top-32 and defer=1 top-32. There were
31,807 baseline-top1-to-candidate pair comparisons. All captured logits were
finite.

| Metric | p50 | p95 | p99 | p99.9 | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| Absolute pairwise margin distortion | 0 | 0.0625 | 0.125 | 0.25 | 0.375 |
| Full-logit max absolute error per position | 0 | 0.256641 | 0.534767 | 0.811062 | 0.8125 |
| Full-logit relative L2 | 0 | 0.00947017 | 0.0224664 | 0.0415018 | 0.0444545 |

The predeclared selection rule in `10_DEFERRED_NUMERICAL_PROTOCOL.md` was
`1.25 × QD max`, not a bound selected after seeing a particular flip. It gives:

```text
B_pair(defer=0 vs defer=1) = 0.46875
full-logit max-abs bound = 1.015625
full-logit relative-L2 bound = 0.05556806921958923
candidate K = 32
```

Under this **QD-only candidate**, 855/855 pairwise-stable positions kept the
exact top-1; 169/169 ambiguous positions selected a token inside the
baseline ambiguity set; pairwise and full-logit envelope overflow were zero.
Raw top-1 agreement was 1023/1024. The one difference was `qd_zh_01` token
9, baseline token 1059 versus defer=1 token 5536, at baseline pairwise
margin 0 and pairwise distortion 0.0625. It is not used to set the bound.
Ambiguous-position ratio 16.5% and maximum ambiguity cardinality 7 are
diagnostic review items, not automatic correctness failures. The candidate
retains an 83.5% stable region.

The machine-readable frozen candidate is
`evidence/qd/PAIRWISE_CONTRACT_CANDIDATE.json`, canonical payload SHA
`d5b228befe5bfc08554b7f9d1c87b4ff6d2348913d2cc94d73a9b3710ca27f26`.
This is **separate from and does not change** the Round 4A/Round 5 retained
defer=0 contract `B_pair=2.1875, K=32`.

## CPU-not-hit control interpretation

The target-row CPU-hit count alone is not a valid whole-request exact-control
predicate. For example `qd_zh_04:0` has zero CPU hits in the target row but
three CPU routes in its prompt-prefix rows. Reconstructing full requests from
the raw JSONL trace and per-request record counts showed that all 1024 QD
requests had at least one CPU route somewhere in their full prompt/prefix.
Thus the strict whole-request CPU-not-hit control was **NOT_OBSERVED**, not
failed. The report is `evidence/qd/full_request_cpu_not_hit.json`; it must
not be quoted as a PASS. An independent zero-CPU-route control remains a
required D1 gate.

## Frozen artifacts

| Artifact | SHA256 of file bytes |
| --- | --- |
| `evidence/qd/defer1_manifest.json` | `db8a25e6698dfee9ec1f73670eb483ab9d1b36977410af3468beec2df284d025` |
| `evidence/qd/defer1_logit_sha256.txt` | `4c32f5c4f31c5dd753e1f3d14dc4bf19b97decea01878112e5f0b58bdc39a3a4` |
| `evidence/qd/matched_metrics.json` | `6f8c3eef5721f481b87412d2a8fde2c805a1f553eb2c4826a060010f365222c2` |
| `evidence/qd/PAIRWISE_CONTRACT_CANDIDATE.json` | `3f178261a1ee71cb626196b89cb811d7af87f4965c5b886ae2c66af05b6a0a5b` |

Raw trace file SHA256: defer=0
`989a627ebc64325c5051ca544d9eb922a40982b023db629afc63588fdaf7bd2c`;
defer=1
`08d62043fb837d982bad05fb6e318b568a89079172e56543904099abe3fcdd1c`.

Next: commit this candidate **before** running HD. HD is used only once for
held-out validation and will not refit `B_pair`, K, or the full-logit bounds.
