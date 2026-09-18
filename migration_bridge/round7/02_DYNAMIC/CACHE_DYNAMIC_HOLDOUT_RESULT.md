# Runtime-v8 Dynamic1024 held-out qualification

2026-09-18 A3 workerB/NPU6, CPU64–127, NUMA2/3, C64=2×32. Frozen
candidate commit0be78ade3835d7adadc2f984fc39e63dd817af12; no threshold refit.

128/128 requests×16 tokens PASS, finite logprobs, COMPLETE/reaped.
Independent local audit re-associated every raw forward with frozen input and
returned history: raw2176, retained2048, knownunused128, unknown0. Recomputed
decode routing and request/update correlation exactly matched summary.
Verified all migration checkpoint-source/weight/mask/generation identities.

Actual triggering32/128=25%, layer updates832, copied slots6656,
changed resident experts4228, migration bytes115,158,810,624.32 triggered
requests provide actual migration evidence, unlike zero-triggerCMMLU.

## Actual separate-run decode comparison

| Metric | Static Uniform | Dynamic1024 |
|---|---:|---:|
|Assignments|299520|299520|
|CPU hits|263230|263938|
|NPU hit rate|12.11605235%|11.87967415%|
|CPU route rate|87.88394765%|88.12032585%|
|CPU weight mass|88.30791403%|88.80098845%|

CPU route count relative delta+0.2689663%. No placement benefit demonstrated
on this16-token decode held-out workload. This does not select a new threshold
or skip frozen longer workload performance qualification. The separate-run
Static comparator is NOT Dynamic's initial-mask counterfactual on Dynamic routes.

Observer-on/concurrent request sum3220.907003s; lifecycle3385.313134s (~56.42min).
These diagnostic timings include exact-weight readbacks and are NOT formal
performance or speedup evidence.

Raw archive `raw/round7-cache-dynamic-holdout-v1.tar.gz`, SHA256
`76b2ba7966a123c8de0f36edbfb7de5fa41932c0ee0e35f1857df96d161affb4`.
Transfer/hash/local full audit PASS. CMMLU174/256invalid0 alreadypassed with
explicit zero-trigger limitation. Next: isolated observer-OFF36-cellABBA;
retain Uniform Static default until performance/default-selection gates close.
