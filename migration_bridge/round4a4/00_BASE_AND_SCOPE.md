# Round 4A.4 Base and Scope

## Repository freeze

Evidence state: **CODE_INSPECTED**.

- Round4A3 parent final: `7b9de4a8249dacfd442f9fb465e2dac6c611f986`
- Round4A4 parent base: `7b9de4a8249dacfd442f9fb465e2dac6c611f986`
- parent branch: `feature/kt-round4a4-pairwise-margin-qualification`
- Round4A3 SGLang final: `8a4bb4c850860377d9880b8c0fa570ca2d0d19e1`
- Round4A4 SGLang base: `8a4bb4c850860377d9880b8c0fa570ca2d0d19e1`
- child branch: `feature/kt-ep-round4a4-pairwise-margin-qualification`

The Round4A3 decision remains `REJECTED`. Round4A4 neither edits its evidence
nor reuses its held-out H corpus for fitting.

## Frozen runtime

- DeepSeek-V2-Lite, TP=1, BF16, batch=1, greedy;
- Layer 17 CPU experts `{6,8,25,36}` and remaining 60 experts on NPU;
- Graph, Deferred Experts, Dynamic Placement, MTP and speculative decoding OFF;
- LLAMAFILE / CPUInfer CPU expert backend;
- one Ascend NPU only (device 0).

## Scope discipline

No GEMM/backend search, placement change, performance tuning, quantization or
TP2 work is authorized. The only child-source change is default-off numerical
instrumentation that captures the already-computed pre-merge NPU partial. It
does not change routing, arithmetic, synchronization, ownership or runtime
behavior when dumping is disabled.

All A3 work runs in disposable container `kt-r4a4-pairwise`, restricted to
CPU cores 0-15 and `/dev/davinci0`. Model/source host mounts are read-only.
Container-local packages and artifacts do not modify the host or business
containers.
