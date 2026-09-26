# Round11 current execution

Updated: 2026-09-26 UTC

Status: `ROUND11_HYBRID_PERFORMANCE_AND_BACKEND_QUALIFIED`

Round11 T0–T6 is complete. The final configuration is the original Round10 HFINAL runtime: request-total-token threshold 129 (`<129` Hybrid, `>=129` Layerwise), Decode Hybrid, Kunpeng C128 with four 32-worker pools on NUMA 2/3/4/7, read-only MXFP4 bank, Static Q32 BF16/NZ29 resident experts, chunk32, PLE ASYNC, Dynamic OFF, original QSA, original Layerwise shadow32, TP1, and serial single-request execution. No Round11 performance candidate is enabled.

T1 keeps the default threshold at 129 and expands experimental Hybrid evidence. Prompt lengths through 768 and Decode through 256 steps are qualified under the frozen single-request contract. Prompt1024 was stopped by the resource guard; prompt769–1023 were not tested and remain unqualified. The expanded-path crossover was not found within the qualified range.

T2 decomposed the frozen three-token Kunpeng replay group: baseline parent median 375.5825 us, attribution median 366.8575 us, native parallel compute plus merge 244.7 us (66.70%), and queue admission 4.4 us (1.20%). Worker runtime max/min was 1.010896. PMU/uncore LLC, bandwidth, and remote-NUMA counters remain UNKNOWN because `perf_event_open` was denied or no validated event mapping existed. T3 selected no bounded single-variable candidate; original C128/MXFP4 remains active.

T4 completed Static/Dynamic four-process evidence. Static steady Decode was 0.3964049145 s/token; Dynamic was 0.4052964265 s/token; exposed migration was 13.824945036 s. Since Dynamic steady state is slower, disposition is `NO_FINITE_BREAK_EVEN`; transaction correctness remains qualified and Dynamic stays OFF.

T5 ranked QSA as the B-route hotspot at 53.87% median attributable request share. GDN was 19.07%; PLE consumer and exposed-wait shares were 0.278% and 0.0177%. The query-aware vectorized sparse-GQA candidate passed the real-device component gate, including maximum relative L2 `5.645496551606974e-05`, large prefill component reductions, neutral Decode, repeatability, and allocator cleanup. It did not reach a request: v145 failed the old launcher source gate before model start, and v146 loaded the model but failed an independent runtime source gate before readiness. Those immutable failures are integration-contract failures, not numerical rejection. The T5 model budget is exhausted, so the candidate closes `COMPONENT_ONLY_NO_PROMOTION` and original HFINAL QSA remains selected.

T6 v147/v148/v149/v150 all passed in BASELINE-A / FINAL-A / FINAL-B / BASELINE-B order. The labels use the exact same HFINAL compute identity. All four processes produced exact matching output IDs for input128 Decode32/64/128, finite logprobs, Hybrid events, and complete cleanup. Five deterministic canary classes passed across the blocks; v147/v149 also passed mid-prefill cancel and same-process reuse.

Unprofiled baseline/final medians were:

| Decode steps | Baseline median | Final median | Final / baseline | Interpretation |
|---:|---:|---:|---:|---|
| 32 | 27.753959 s | 27.584503 s | 0.993894 | Same-code process variance |
| 64 | 40.775423 s | 38.996865 s | 0.956382 | Same-code process variance |
| 128 | 71.157255 s | 68.513867 s | 0.962851 | Same-code process variance |

The T6 peak across independent blocks was 28,783 MiB whole-card HBM, 25,766 MiB owned-process HBM, and 336.194 GiB host cgroup. Each block used a fresh two-round admission. After evidence hashing and process cleanup, the exact 179-file project cache hook audited `/proc/*/fd` and mmap references, called only `POSIX_FADV_DONTNEED`, preserved file identities, and verified zero resident pages. No global cache clear or external-task change occurred.

Final budget is 19/21 model starts, 80/108 generation attempts, 4/5 abort controls, and zero automatic retries. T2 used three component processes; T5 used three. Physical12 is empty, no Round11 container or process remains, and automation remains paused. No push, merge, tag, reset, or clean was performed.

Primary evidence:

- `ROUND11_FINAL_STATUS.json`
- `ROUND11_T6_FINAL_ANALYSIS.json`
- `ROUND11_T5_FINAL_DISPOSITION.json`
- `ROUND11_RESOURCE_RETIREMENT_RECEIPT.json`
- `ROUND11_EXECUTION_BUDGET.json`
- `EXECUTION_LEDGER.md`
- `docs/reports/2026-09-26-round11-final-zh.md`
