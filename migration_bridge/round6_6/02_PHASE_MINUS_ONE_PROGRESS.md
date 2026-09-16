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

## Native stages and layout controls

Stage replay v2 using diagnostic build v3:512/512 production-output parity, all stages finite. Maximum production-vs-FP32 relL2 gate0.0017732126, up0.0017981203, mul0.0040710656, down0.0044835914. Maximum production-vs-GMM relL2 gate0.0004198827, up0.0002579507, mul0.0005302370, down0.0015245553.
Oracle-layout comparison:512/512 one-slot vs64-slot NPU outputs exact. This closes the layout caveat for those frozen operands, not arbitrary untested shapes.

Build v2 is not used as evidence: it was launched before corrected-header delivery completed. Build v3 was launched after explicit source SHA verification (`a3b2c34cf653c9194dfe3a7a10da12ad1181702648bb994add45c9f7fb084fe3`) and is the admissible diagnostic extension. Original production installation remains untouched.

## Expanded real prefill

Operator corpus SHA:`07d380b85dad0dfe6ac40b7501fc7836bae6116396e15976ceb6301db3aa96e7`.
Real capture:3 requests,6 audited forwards; prompt lengths16/64/128. No QD6/HD6 output.
Replay v1 failed safely at qlen64: harness capacity16 triggered `_check_qlen_fits_cpp_buffers`, before native output-buffer overflow. This is a harness capacity mistake, not production arithmetic failure. Recovery v2 allocates capacity from the maximum frozen qlen128; keeps the same selected experts, source operands, workers, and arithmetic. Retain v1 failure and rerun96 cases in fresh output roots.
