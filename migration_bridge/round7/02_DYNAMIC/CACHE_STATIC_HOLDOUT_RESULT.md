# Runtime-v8 Static Uniform HOLDOUT

2026-09-18 A3, workerA/NPU4, CPU0–63, NUMA0/1, C64=2×32. DynamicOFF,
DeferredOFF, same immutable model/GGUF and Uniform R8 profile. Concurrent
independent Dynamic HOLDOUT: timings below are descriptive, not formal ABBA.

Result PASS:128 unique requests×16 generated tokens, finite output logprobs,
COMPLETE lifecycle and owned reaping. Exact raw request SHA matched summary.
Forward attribution raw2176, retained2048 (prefill128/decode1920), known final
unused-token forwards128, unknown0. Retained/unused boundaries not discarded.

Combined prefill/decode routing:19,972,992 assignments; NPU2,484,750,
CPU17,488,242; NPU hit12.4405497%, CPU route87.5594503%, CPU weight mass87.0509985%.
Combined metrics must not be substituted for decode-only causal comparison.

Request-time sum2094.078304s, lifecycle2251.152023s (~37.52min).
No formal performance claim due observer overhead and concurrent jobs.

Archive `raw/round7-cache-static-holdout-v1.tar.gz`, SHA256
`38cee92cfa74ec2af8d8a6893d56198cbf7a80943525803709d1abb89d565a74`.
SCP/hash verified, local raw count/finite/hash/lifecycle/attribution audit PASS.

Dynamic HOLDOUT remains pending (106/128 at06:09). Do not start formalABBA yet.
Before formal performance, start a **new campaign** of unchanged host observer
source: old monitor-v3 compares swap counters against its own long-lived start,
so historical swap activity latches WAIT even when currently quiescent. Preserve
v3 historical data and criteria; do not relabel old windows or weaken checks.
Fresh campaign must show actual no-contender/zero-owned-swap/no-active-host-swap
and60s quiet admission, with continuous measurement-window evidence.
