# Round15 Decode batch-1 segmented NPU graph contract

Status: `P0_CONTRACT_FROZEN_IMPLEMENTATION_NOT_RUN`

## Boundary

The first B implementation may capture only one layer's existing resident-NPU
MoE application:

```text
graph outside: route + physical-ID remap + CPUInfer submit
graph input:   hidden[1,2560], ids[1,10], weights[1,10]
graph body:    npu_moe_init_routing_v2
               -> grouped BF16 gate/up matmul
               -> npu_swiglu
               -> grouped BF16 down matmul
               -> npu_moe_finalize_routing
graph output:  resident contribution[1,2560]
graph outside: CPU result wait + FP32 merge + remaining layer
```

Prefill, batch sizes other than one, empty/unsupported route geometry, Layerwise
mode, CPUInfer, PLE host gather, request scheduling, cancellation decisions,
GDN/QSA state mutation and HTTP handling remain eager.

## Static storage and live data

- Each graph owns fixed input and output buffers.  Replay copies the current
  token's hidden state, remapped physical IDs and router weights into those
  buffers before replay.
- Per-layer resident weights retain their production addresses throughout graph
  lifetime.  A weight or mapping publication retires the graph before the new
  identity can execute.
- Capture warmup uses scratch tensors and the resident MoE operation only.  It
  cannot advance KV, GDN, QSA or PLE state.
- Graph output is copied/consumed only after replay completion on the existing
  dependency stream.  Existing CPU-result and merge fences remain.

## Cache identity and caps

The cache key is:

```text
(layer_id, segment='resident_moe', implementation_sha256,
 weight_storage_identity, resident_mapping_generation,
 dtype='bfloat16', weight_layout='NZ29', batch=1, topk=10,
 hidden=2560, stream_identity, storage_binding_version)
```

Request epoch is replay data and does not create a new graph.  Initial scope is
one graph per routed layer, at most 48.  The hard contract remains at most two
segments per layer and 96 active graphs.  Static buffers, graph pools, bridge
buffers and operator workspaces together must add no more than 4 GiB and must
also remain below the stricter Round14a allocator/owned/board caps.

## Eligibility and eager fallback

Fallback is selected before capture/replay when any of the following differs:

- mode is not Decode;
- batch is not one;
- hidden/top-k geometry, dtype or layout differs;
- resident mapping or weight storage identity differs;
- graph is absent, retired, failed, or exceeds memory/instance caps;
- cancellation/close has started;
- the local torch-npu/CANN capability probe did not qualify every graph op.

An eligibility fallback has no graph side effect and may execute eager.  A
replay failure is terminal for that request: stop new replay, drain the stream,
retire the graph and close the request.  The same token must not be run eagerly
unless a separate rollback test proves that no state/output was committed.

## Lifecycle

1. **Create:** validate source/binding/weights, allocate bounded static buffers,
   run eager scratch warmup, capture once, restore/verify scratch state, and
   register the immutable key.
2. **Replay:** verify request and mapping epochs, copy current tensors, record
   input-ready, replay on the bound stream, record completion, publish output,
   then allow the existing CPU/NPU merge.
3. **Cancel:** mark request closed, refuse new replay, wait for in-flight graph
   completion, discard unpublished output, then release request leases.
4. **Retire:** wait for all replays, reset the NPUGraph, release static buffers,
   unregister the key and record allocator/owned memory.  Close and model unload
   must retire all graphs.

## Evidence required before request testing

- device capability and graph-op capture receipt;
- one real-layer eager/capture/replay numerical fixture with changing hidden,
  IDs and weights across at least 20 replays;
- eager and replay output hashes/relative errors, unsupported-shape fallback,
  cancel/retire and replay-failure behavior;
- no per-request graph growth and no per-replay allocated/reserved growth;
- two paired blocks where the complete Hybrid layer cost improves at least 10%.

Only then may B run two model smoke processes and the four-process E/G/G/E
request comparison.  Replay success or reduced Python submission time alone is
not promotion evidence.
