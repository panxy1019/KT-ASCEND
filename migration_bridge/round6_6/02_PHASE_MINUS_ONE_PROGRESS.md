# Phase -1 progress (not B0 closure)

## Output screen v1

128/128 CPU-owned expert instances, 384 decode samples +128 real qlen11 prefill samples =512 cases. Each production/NPU comparison repeated10 times. All deterministic and finite; no local 0.01 safety overflow.

| Relative L2 | Median | Maximum |
|---|---:|---:|
| LLAMAFILE vs pure CPU FP32 | 0.0033454691501853162 | 0.004483591392794621 |
| NPU GMM vs pure CPU FP32 | 0.003345015027797474 | 0.004482310916015284 |
| LLAMAFILE vs NPU GMM | 0.000008678345546625814 | 0.0015245552599669338 |
| LLAMAFILE vs CPU BF16-visible reference | 0.000023998445161716174 | 0.0018443704150615185 |

Error ratio LLAMAFILE/reference over NPU/reference: min0.9907693612393256, median1.0, max1.0094429545035501. These are descriptive, not a new fitted bound.
Scope caveats: isolated one-expert physical NPU layout; actual deployed GMM method, not F.linear. Additional equivalence with production-sized layouts remains to audit. Real inputs used for unrouted experts are explicitly labelled controls. One prefill bucket is not comprehensive prefill coverage.

## Native stage capture v1 failure

After three passing decode observer-parity cases, qlen11 entered tail-safe sequential forward_one calls. The diagnostic file writer used exclusive creation per expert/stage, which incorrectly rejected the second row. Error: R66_STAGE_CAPTURE_OPEN_FAILED; guard EXIT_NONZERO (-6); all owned descendants reaped. This was a newly introduced observer defect, not evidence of a production numerical defect.

Recovery v2: append rows within the harness's newly created exclusive per-forward directory; exact raw float count/shape is checked. Rebuild only test bindings, never replace the installed production extension. Retain v1 files and original failure; use separate v2 build/result directories.

B0 remains IN_PROGRESS. QD6/HD6 remain NOT_RUN.
