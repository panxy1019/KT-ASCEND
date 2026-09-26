# H0 selection and state coupling graph

The statements below come from the exact v90 snapshot and its run receipts.

```text
plain scheduler batch
        |
        +-- forward_mode != EXTEND ------------------------------> Hybrid KTEP
        |                                                          (Decode included)
        |
        +-- EXTEND, serial request
                 |
                 +-- len(origin_input_ids) < 129 ----------------> Hybrid KTEP
                 |       |
                 |       +-- CPU MXFP4 submit/native workers
                 |       +-- static Q32 NPU BF16/NZ29 residents
                 |       +-- sync and merge
                 |       +-- no Round9 route-collector snapshot
                 |
                 +-- len(origin_input_ids) >= 129 ---------------> Layerwise
                         |
                         +-- each scheduler chunk must be 1..32
                         +-- original shadow32 transport
                         +-- route collector observes all 48 layers
                         +-- request completion may call
                             PrefillEndBoundary.prefill_complete(stats)
```

The threshold is the request's full `origin_input_ids` length after the actual
template/tokenizer, not the current dispatch row count. The comparison is
`total < threshold -> Hybrid`; therefore Layerwise begins at 129. The check is
inside `PrefillRuntime.before` and first rejects non-EXTEND forward modes, so
Decode does not accidentally enter Layerwise.

The framework and streaming profiles must carry the same threshold. Streaming
splits the selected Layerwise request into chunk32; it does not change the
selection basis. Resident Q32, private shadow32, and scheduler chunk32 are three
different capacities.

H0 has `route_collection=false`. Hybrid still executes the real router and the
KTEP CPU/NPU branches, while the Round9 request-completion statistics used by
the placement boundary are created only by the Layerwise collector. In this
snapshot `prefill_after` forwards statistics to `PrefillEndBoundary` only when
`PrefillRuntime.after()` returns a completed Layerwise snapshot. Dynamic is
also explicitly disabled by both the server argument and boundary environment
flag. T4 must therefore establish a named Hybrid-to-Dynamic statistics/trigger
contract rather than claiming v90 already exercises it.

PLE ASYNC, the static uniform Q32 resident map, the BF16/NZ29 NPU source, and
the read-only MXFP4 CPU bank do not change when the policy selects Hybrid or
Layerwise. T1 must verify the real Hybrid branch events rather than infer them
from selection alone.

Evidence entrypoints:

- `h0-v90-source/overlay-python/sglang/srt/layers/round9/prefill_runtime.py`
- `h0-v90-source/overlay-python/sglang/srt/layers/round9/prefill_binding.py`
- `h0-v90-source/overlay-python/sglang/srt/layers/round9/streaming_binding.py`
- `h0-v90-source/overlay-python/sglang/srt/layers/round9/hybrid_probe.py`
- `h0-v90-source/overlay-python/sglang/srt/layers/moe/kt_ep_wrapper.py`
- `../round9-fixed-shadow32-update-20260925/v90/run/{server,framework,streaming}-profile.json`

