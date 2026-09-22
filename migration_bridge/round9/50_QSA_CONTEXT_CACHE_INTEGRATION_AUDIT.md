# F8/F9/F10 QSA context-cache integration audit

Date: 2026-09-21  
Scope: read-only audit of the local F8 context contract and the preserved Qwen4/SGLang-v7 source on `admin@110.123.0.3:/mnt/data1/kt-round9/framework-sglang-v7`. No NPU process, container, or remote file was changed.

## Decision

The existing allocator CPU plan correctly accounts for its supplied geometry, but its test fixture uses synthetic GDN/PLE byte values and is not a deployment budget or an implementation seam for context tiering. The current Qwen4 reference path requires every historical ordinary K/V row to remain in the NPU pool, and the compressed QSA index is addressed through the same physical page number. Calling the existing allocator's `free()` after copying a page to host would therefore leave both `req_to_token` and the QSA compressed-index lookup pointing at storage that can be reassigned.

The smallest credible F9/F10 route is a Qwen4-only, request-local tier with three explicit changes:

1. keep the QSA index in BF16 and address completed groups by `(request identity, logical group, layer)`, independent of physical KV page numbers;
2. represent completed ordinary-KV pages in a request-local logical page table, with either a resident physical page or a canonical host FP8 record;
3. after the indexer selects logical token IDs, gather only the selected pages into a layer-scoped BF16 staging tensor, remap selected logical IDs to staging rows, and call the existing `sparse_gqa` numerical consumer.

GDN and PLE remain in their current request-owned recurrent pools for the whole request. Prefix/radix sharing remains disabled. The first implementation must not enable the global native FP8 KV-cache option: the NPU pool can store float8 bytes, but the preserved Qwen4 path has no qualified scale, offload, or FP8-consumer contract.

## Audited geometry

The checkpoint at `/home/admin/models/Qwen3.8-Flash-Next/de4b8e4d43b917e7706784d8bb445c9af86a3540/config.json` has 48 layers with the repeating `linear, linear, linear, full` pattern: 36 GDN layers and 12 QSA layers. It declares 2 KV heads of width 256, QSA index width 128 with one index KV head, compression ratio 4, index budget 2,048 tokens, one PLE layer, hidden size 2,560, `hc_count=4`, PLE convolution kernel 4, and n-gram size 3.

The local `runtime/context_profile.py` correctly fixes the supported targets to 4K/16K/64K/128K (line 10), requires BF16 ordinary KV and index geometry with ratio 4 (lines 58-78), calculates ordinary K/V and compressed-index bytes separately (lines 100-138), and binds distinct QSA_KV, QSA_INDEX, GDN, and PLE identities to one request/prefix/allocation epoch (lines 163-195). The ownership helpers are logical planning objects only (lines 198-222); they do not alter SGLang allocation.

All admitted Round9 profiles and the current physical15 run use TP1. The active profile `runtime/profiles/q64_bf16_c64_sync_physical15.json` fixes `tp_size=1`, `max_running_requests=1`, `max_total_tokens=4100`, `context_length=4098`, `max_mamba_cache_size=1`, and page size 4. The v32 server receipt confirms the resolved values `tp_size=1`, `max_total_num_tokens=4100`, and `context_len=4098` in `evidence/physical15/20260920-full-model-v32/server.log`.

The local `test_context_profile.py` geometry uses correct QSA dimensions but deliberately supplies `gdn_bytes_per_request=4096` and `ple_bytes_per_request=64` (lines 13-30). Its `ContextPlan.total_bytes` values below are therefore **synthetic fixture totals**. They must never be reported as a real Round9 memory budget:

| Context | Token capacity | Logical pages | BF16 ordinary QSA K/V | BF16 QSA index | Synthetic fixture total |
|---:|---:|---:|---:|---:|---:|
| 4,096 | 4,100 | 1,025 | 100,761,600 B | 3,148,800 B | 103,943,280 B |
| 16,384 | 16,388 | 4,097 | 402,751,488 B | 12,585,984 B | 415,419,504 B |
| 65,536 | 65,540 | 16,385 | 1,610,711,040 B | 50,334,720 B | 1,661,324,400 B |
| 131,072 | 131,076 | 32,769 | 3,221,323,776 B | 100,666,368 B | 3,322,530,928 B |

At TP1, the checkpoint's GDN formula in `configs/qwen3_next.py:284-303` and `configs/mamba_utils.py:141-176` is exact:

- projected value width: `48 value heads * 128 = 6,144`;
- convolution input width: `6,144 + 2 * 16 key groups * 128 = 10,240`; TP divisor 1 leaves 10,240;
- convolution state: 36 layers x `[10,240, 3]` BF16 = 2,211,840 bytes;
- recurrent state: 36 layers x `[48, 128, 128]` FP32 = 113,246,208 bytes;
- total live GDN state per request: **115,458,048 bytes**.

The PLE shape in `configs/qwen4_exp.py:101-119` is `[10,240, 9]` BF16 plus two int64 n-gram history entries, or **184,336 live bytes per request**. `mem_cache/memory_pool.py:484-495` confirms that the PLE convolution uses the GDN convolution dtype and registers both PLE pools under the recurrent slot. These are lifecycle state, not token-proportional spill candidates. A device gate must confirm the calculated shapes because environment dtype overrides are permitted by `configs/mamba_utils.py:47-107`.

Replacing the fixture values with those checkpoint-derived TP1 live-state bytes gives the following logical one-request budgets. The total retains the planner's 16-byte-per-logical-page accounting, so it remains a planning total rather than a measurement of every runtime tensor:

| Context target | Capacity (`target+4`) | QSA K/V | QSA index | TP1 GDN | PLE | Derived live planning total | Known physical tensor floor |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4,096 | 4,100 | 100,761,600 B | 3,148,800 B | 115,458,048 B | 184,336 B | 219,581,504 B | 335,308,864 B |
| 16,384 | 16,388 | 402,751,488 B | 12,585,984 B | 115,458,048 B | 184,336 B | 531,057,728 B | 646,735,936 B |
| 65,536 | 65,540 | 1,610,711,040 B | 50,334,720 B | 115,458,048 B | 184,336 B | 1,776,962,624 B | 1,892,444,224 B |
| 131,072 | 131,076 | 3,221,323,776 B | 100,666,368 B | 115,458,048 B | 184,336 B | 3,438,169,152 B | 3,553,388,608 B |

Runtime allocation includes padding beyond that logical table. `NPUMHATokenToKVPool._create_buffers()` (`hardware_backend/npu/memory_pool_npu.py:51-70`) allocates `size/page_size + 1` physical pages: with `size=4100`, page 0 is padding and pages 1-1025 are usable, for 4,104 physical token rows. `Round9QSAKVPool` similarly allocates `(size+page)/4=1026` compressed groups (`round9_qsa_pool.py:24-34`), one more than the 1,025 logical groups, while its raw ring has one request slot. Separately, `MambaPool` and both PLE pools allocate `size+1` state slots (`memory_pool.py:245-264`; `ple_state_pool.py:75-79,164-169`): with `max_mamba_cache_size=1`, slot 0 is padding and slot 1 is the sole usable request slot. Thus the admitted 4K process allocates two GDN/PLE state slots even though only one can be live.

For the admitted 4K profile, the directly derived context-tensor allocation floor is therefore **335,308,864 bytes**: 100,859,904 ordinary-KV bytes (4,104 rows), 3,151,872 compressed-index bytes (1,026 groups), 12,320 raw-ring/position bytes, 230,916,096 GDN bytes (two slots), and 368,672 PLE bytes (two slots). This excludes `req_to_token`, allocator objects, framework allocator overhead, QSA staging, and the planner's abstract page metadata; the later device receipt must measure those separately. The profile's four capacity slots above target 4,096 are also not all user context reserve: `context_length=4098` leaves two token rows beyond admitted context, while the physical page-0 padding adds four more non-usable rows.

The CPU implementation now exposes these distinctions directly. `derive_qwen4_tp1_geometry(config, config_scope=...)` requires the caller to identify a full config or a text config, validates the checkpoint's 48-layer 3GDN:1QSA pattern and all dimensions used by the derivation, and calculates GDN/PLE bytes from those fields. `derive_qwen4_tp1_footprint()` preserves the generic `ContextPlan` while adding logical capacity/pages, the physical padding page, raw request slots, usable recurrent slots, padded recurrent storage slots, and eight exact tensor expectations. `validate_qwen4_tp1_measurements()` accepts only caller-supplied shape/dtype/nbytes records and requires an exact match; it imports no device runtime and makes no deployed claim. Its scope is `CPU_DERIVED_TP1_PHYSICAL_FLOOR_NOT_DEVICE_MEASURED` until a later runtime records the actual tensors.

## Actual consumer and storage seams

### QSA index and ordinary K/V

`models/qwen4_exp.py:365-463` is the complete index path. The pending four-token raw-key ring is request-slot addressed (`:399-406`). On each completed group the code reads the first ordinary-KV physical slot from `req_to_token` and writes the compressed key at `physical_slot // 4` (`:430-439`). Selection reconstructs every logical group start, maps it back through `req_to_token`, gathers compressed keys by physical page, and returns logical token IDs (`:441-463`). Thus the compressed index cannot survive freeing and reusing an ordinary KV page in the present layout.

`models/qwen4_exp.py:621-667` is the ordinary-KV consumer. It writes new K/V through `set_kv_buffer` (`:634-641`), obtains selected logical IDs from the indexer, then reads the entire logical range from `req_to_token`, rejects any non-positive entry, gathers every K/V row from the layer pool, and only then passes the full tensors plus selected IDs to `sparse_gqa` (`:642-664`). This is the exact seam that must change. Host pages or negative/sentinel mappings cannot feed the current code unchanged.

`layers/round9/qsa.py:22-49` selects logical token IDs from BF16 compressed keys using FP32 scoring. `sparse_gqa` at `:52-71` only requires query-shaped data, K/V rows, and indices into those rows; it gathers selected rows and performs the reference math in FP32. Therefore it can remain the numerical oracle if the tier supplies a compact BF16 K/V tensor and remapped indices.

### Pool and allocator

`mem_cache/round9_qsa_pool.py:12-35` wraps `HybridLinearKVPool`, allocates raw and compressed QSA index tensors with the **same dtype as the ordinary KV pool**, and sizes compressed storage by physical token capacity. Its get/set API (`:37-61`) has no logical request/group identity. Its `get_contiguous_buf_infos()` explicitly rejects disaggregation/offload (`:63-64`).

`model_executor/model_runner_kv_cache_mixin.py:977-1008` is the construction seam for `Round9QSAKVPool`; `:1051-1067` installs `NPUPagedTokenToKVPoolAllocator`. A tier manager should be bound here after both pool and allocator exist, under the existing Round9 profile checks. It should not be installed in generic MHA/MLA configurations.

`hardware_backend/npu/allocator_npu.py:28-94` allocates whole free pages for extend and `:96-133` allocates a page at decode boundaries. `free()` at `:135-148` returns the unique page numbers to the global free list. It has no resident/spilled state. `mem_cache/allocator.py:354-400` documents the page-aligned contract; its CPU-copy methods merely delegate to the KV-cache wrapper (`:531-535`).

The generic CPU-copy implementation in `mem_cache/memory_pool.py:927-960` synchronizes CUDA, copies every layer, and is not an NPU request-local path. More directly, `HybridLinearKVPool` delegates K/V get/set to its inner full pool (`:1222-1388`) but does not delegate `get_cpu_copy` or `load_cpu_copy`; the base methods at `:726-730` raise `NotImplementedError`. Existing `Req.offload_kv_cache()` in `managers/schedule_batch.py:1174-1185` therefore is not usable for Qwen4. Its retraction lifecycle also clears `mamba_pool_idx` (`:1141-1172`), which violates the requirement to preserve GDN and PLE state.

With radix disabled, allocation pressure cannot evict a useful prefix: `mem_cache/common.py:255-294` asks the tree cache to evict before allocator extend, but the disabled `ChunkCache` is a no-op cache. This is desirable for the profile boundary, but it means a real tier must return physical pages to the allocator itself and own the corresponding logical mappings and release path.

## FP8 capability: what exists and what is missing

There is a native **storage representation**, not a qualified Qwen4 FP8 cache path:

- `mem_cache/memory_pool.py:653-661` stores `float8_e4m3fn`/`float8_e5m2` buffers as `uint8` because float8 indexed writes are unavailable.
- `hardware_backend/npu/memory_pool_npu.py:51-70` allocates the contiguous paged NPU buffer with that storage dtype.
- Its `set_kv_buffer()` at `:112-164` optionally divides by caller-provided scales, casts to the configured float8 dtype, reinterprets bytes, and invokes the NPU scatter/cache write.
- `mem_cache/memory_pool.py:962-985` reinterprets stored bytes as float8 when read.

The current Qwen4 call supplies no scales (`models/qwen4_exp.py:640`), while `HybridLinearKVPool.set_kv_buffer()` defaults both scales to 1.0 (`memory_pool.py:1359-1378`). Worse, `Round9QSAKVPool` gives the QSA raw and compressed index the global KV dtype (`round9_qsa_pool.py:24-30`), so enabling global FP8 would silently quantize the selector's index as well. There is no per-page scale metadata, host-page format, NPU offload implementation, Qwen4 numerical error receipt, or proof that all operations used by the eager consumer accept these tensors. The allocation/write machinery alone is insufficient to mark native FP8 qualified.

The initial F9 reference route should therefore keep the active pool and QSA index BF16 and use a software-defined host record:

```
PageKey = (request_id, allocation_epoch, logical_page)
LayerRecord = {
  layer_id,
  valid_tokens,              # 1..4 for tail, normally 4 before spill
  shape = [valid_tokens, kv_heads_on_this_rank, 256],
  k_format = v_format = "float8_e4m3fn",
  k_scale_f32, v_scale_f32,
  k_bytes, v_bytes,
}
PageRecord = {PageKey, state, layer_records[12], sha256}
```

Use one finite positive FP32 scale per `(page, layer, K-or-V)` for the first bounded implementation. Derive it from `amax / torch.finfo(torch.float8_e4m3fn).max`, with scale 1 for an all-zero tensor; quantize `BF16 / scale` to E4M3FN bytes. Reload by reinterpreting the bytes, multiplying by the recorded scale in FP32, and converting to BF16. Reject NaN/Inf input, invalid scales, shape/dtype/version mismatch, digest mismatch, and a stale request/allocation epoch before publishing a page as spilled or staged. Per-head scaling may be evaluated later, but changing the format belongs behind a new version and quality gate.

For the admitted TP1 two-head geometry, one four-token page contains 98,304 bytes of BF16 ordinary K/V. Its host FP8 payload is 49,152 bytes plus 96 bytes of scales. Keeping the corresponding QSA index group in BF16 costs 3,072 bytes, for **52,320 bytes per logical page** plus bounded identity/digest metadata. At 128K plus the reserved page this is 1,714,474,080 bytes of canonical host FP8 K/V plus resident BF16 index, versus 3,321,990,144 bytes for those two BF16 components in the current plan.

## Minimal request-local design

### 1. Decouple QSA index identity

Change the compressed-key location from `physical_slot // 4` to a bounded request-logical group location. With `max_running_requests=1`, a tensor `[12, planned_groups, 1, 128]` BF16 is sufficient, but the API must still require the `ContextStateIdentity` and validate the allocation epoch. The raw four-entry ring remains request-local and BF16. Index writes and reads in `Qwen4ReferenceIndexer` then use `logical // 4` and the request's group extent. This preserves exact selection semantics while allowing the ordinary page to move.

Do not spill the compressed index in F9. Selection consults every completed group on every QSA layer, so keeping roughly 100.7 MB at the 128K model geometry avoids an unbounded index reload and makes selection independent of ordinary-KV residency.

### 2. Add a logical page table beside `req_to_token`

For each request identity and logical page, store exactly one of:

- `RESIDENT(physical_page, generation)`;
- `SPILLING(physical_page, transfer_id)`;
- `SPILLED(host_record)`;
- `LOADING(host_record, staging_lease)`.

Only completed four-token pages may spill. Keep the active tail and the immediately preceding page resident until allocation has crossed the boundary; NPU extend/decode uses `last_loc` and its debug checks at `allocator_npu.py:37-45` and `:102-105`. Chunked prefill must reserve enough resident pages for the whole in-flight chunk plus this tail rule before it starts.

After the host record and digest are complete, atomically replace the logical-page entry with `SPILLED`, invalidate that page's ordinary physical mappings for Qwen4 consumers, then return the physical page through the allocator. The generic `req_to_token` table cannot safely contain a freed positive slot. A sentinel is acceptable only after all Qwen4 users are tier-aware: the current model rejects it (`qwen4_exp.py:647-651`), generic NPU block-table construction reads page starts (`hardware_backend/npu/attention/ascend_backend.py:268-279`), and request cleanup currently frees every mapped token. Until those seams are routed through the tier, the gate must assert that no physical page has been returned.

The tier owns final release: wait/cancel transfers, free only currently resident physical pages once, retire staging leases, erase host records and QSA-index extents, then release the existing GDN/PLE request slot. Abort never imports a page or creates a new recurrent slot. `ChunkCache` remains disabled and receives no prefix insertion.

### 3. Feed the real QSA consumer with layer-scoped staging

Replace `models/qwen4_exp.py:647-664` with a Round9 tier API called after `selected` exists:

```
attn_output = tier.sparse_gqa_selected(
    identity, layer_id, q, selected, current_logical_position,
    resident_pool=pool.full_kv_pool,
    reference_consumer=sparse_gqa,
)
```

`sparse_gqa_selected` must:

1. preserve each query row's selected-token order and duplicates;
2. validate causal bounds and identity before any transfer;
3. form the bounded unique set of `(logical_page, offset)` entries selected by the QSA budget plus the at-most-three-token tail;
4. gather resident rows directly and decode spilled FP8 records into a layer-scoped BF16 staging tensor;
5. build a deterministic logical-token-to-staging-row map and remap `selected`, preserving `-1` padding;
6. call the existing `sparse_gqa` on one query row and its compact tensors, and hold the staging lease until that call has finished.

The configured selector has at most 2,048 completed tokens plus a three-token tail per query. A single layer's admitted TP1 two-head BF16 staging bound is `2,051 * 2 * 2 * 256 * 2 = 4,200,448` bytes for K+V per query row before page deduplication. The bounded reference processes query rows serially, matching the row loop already present in `layers/round9/qsa.py:63-70`, so a long prefill chunk does not multiply staging capacity by its number of query rows. A later batched staging optimization must declare a fixed row-concurrency limit and reserve `limit * per_row_bytes` before launch. Staging all 12 layers or all historical tokens defeats the capacity objective and is forbidden by the contract.

Resident and spilled rows may share one compact staging tensor for the reference implementation. This keeps the existing `sparse_gqa` untouched and provides a direct baseline comparison. An optimized consumer can later gather pages directly, but it must pass the same ordered-output gate first.

### 4. Lifecycle boundaries

Spill publication is permitted only after the model write for all 12 QSA layers has completed for that page. Reload publication is permitted only after the full record has passed identity, shape, scale, and digest validation. Transfer callbacks carry `(request_id, allocation_epoch, logical_page, transfer_id)`; stale callbacks discard their result without touching the current mapping.

GDN (`mamba_pool_idx`, 36 conv states and 36 recurrent states) and PLE (short-conv state plus n-gram history) stay allocated and retain pointer/hash identity across every spill/reload. Request end performs one tier release and one recurrent-slot release. It never invokes the disaggregation retraction path and never starts a second migration.

## Required gates

No 64K/128K or native-FP8 claim is earned until all earlier gates pass.

### Runtime pool measurement helper

`runtime/context_measurement.py` now provides the read-only integration API:

```python
measurement = measure_and_validate_qwen4_tp1_pools(
    runner, footprint, require_device='npu')
```

The caller must first derive `footprint` from the trusted checkpoint config with `derive_qwen4_tp1_geometry` and `derive_qwen4_tp1_footprint`. The helper performs no allocation, imports no Torch or device module, and does not search for similar tensors. It follows eight exact live-object paths and returns each path with its measured shape, dtype string, `nbytes`, device string, tensor pointer, and underlying untyped-storage pointer and size:

| Profile category | Exact runtime attribute chain |
|---|---|
| `ordinary_kv` | `runner.token_to_kv_pool.full_kv_pool.kv_buffer` |
| `qsa_compressed` | `runner.token_to_kv_pool.qsa_compressed_keys` |
| `qsa_raw` | `runner.token_to_kv_pool.qsa_raw_keys` |
| `qsa_positions` | `runner.token_to_kv_pool.qsa_rope_positions` |
| `gdn_conv` | `runner.req_to_token_pool.mamba_pool.mamba_cache.conv[0]` |
| `gdn_temporal` | `runner.req_to_token_pool.mamba_pool.mamba_cache.temporal` |
| `ple_conv` | `runner.req_to_token_pool.short_conv_pool.conv_state` |
| `ple_history` | `runner.req_to_token_pool.ngram_pool.context` |

The paths were checked read-only against the preserved v32 SGLang tree at `/mnt/data1/kt-round9/framework-sglang-v7/python/sglang/srt`. That tree is the `PYTHONPATH` recorded by the v32 container. `framework-sglang-v8` has identical SHA-256 values for the four relevant files: `memory_pool.py` `7bf24bd3bb92c0453aa784386a631d89927487000d4bf64f988921641f79813e`, `ple_state_pool.py` `110a33c09c5b2d2699920e4d278d4081d76d6bcd087974608caf22666e297d9a`, `round9_qsa_pool.py` `99ae4f5a3db4af3c1c3fe1e79295dd0c2159fa5427271678b605d7d09c567a7a`, and `memory_pool_npu.py` `5ff48ef9ebcb10e10bd55baa794d1fa8c22071b1e69727c5764681bec20ea181`.

The source facts behind the chains are concrete. `NPUMHATokenToKVPool._create_buffers` creates `kv_buffer` and its K/V views (`hardware_backend/npu/memory_pool_npu.py:51-70`). `HybridLinearKVPool` retains the passed `mamba_pool` and creates `full_kv_pool` (`mem_cache/memory_pool.py:1222-1275`), while `Round9QSAKVPool` creates the compressed, raw, and position tensors (`mem_cache/round9_qsa_pool.py:12-34`). `MambaPool` stores its one admitted conv tensor in `mamba_cache.conv` and its recurrent tensor in `mamba_cache.temporal` (`mem_cache/memory_pool.py:220-264,296-323`). `HybridReqToTokenPool` owns that `mamba_pool` and the two PLE pool objects (`memory_pool.py:443-495`); their backing tensors are `ShortConvPool.conv_state` and `NGramPool.context` (`mem_cache/ple_state_pool.py:42-99,132-184`).

The storage-owner requirements also match these constructors. NPU ordinary KV is one direct `torch.zeros` allocation assigned to `kv_buffer`; only `k_buffer = kv_buffer[0]` and `v_buffer = kv_buffer[1]` are views, and the helper measures the owning `kv_buffer`, not either view (`memory_pool_npu.py:57-70`). Each QSA tensor is a separate direct `torch.zeros` or `torch.full` allocation (`round9_qsa_pool.py:24-34`). Each GDN conv entry and the temporal tensor are separately allocated with `torch.zeros` (`memory_pool.py:245-264`). The only constructor-side conv layout rewrite is `_init_amx_conv_state`, guarded by `_is_cpu and _cpu_has_amx_support` (`memory_pool.py:254-258`); it does not run for the admitted NPU profile. `MambaPool.State.at_layer_idx` later returns per-layer views for model consumers (`memory_pool.py:186-205`) but does not replace the root tensors measured by the helper. PLE conv and history are likewise direct, separate `torch.zeros`/`torch.full` allocations (`ple_state_pool.py:70-79,159-169`). A read-only assignment census in the preserved v7 tree found no later replacement assignment for any of the eight root attributes. Thus contiguous, zero-offset, full-storage and pairwise-distinct-storage checks describe the real admitted constructors rather than imposing a view layout they do not use.

Validation additionally requires that the QSA and request pools reference the same `MambaPool`, the conv list contains exactly one tensor for this admitted checkpoint, QSA layer/request counts and all three recurrent capacities equal the supplied TP1 footprint, and each tensor's `shape` product, `numel()`, `element_size()`, and reported `nbytes` agree internally. Every category must be a contiguous, zero-offset owner whose tensor pointer equals its storage pointer and whose storage size equals its logical `nbytes`; the eight storage pointers must be distinct. This rejects two different view objects backed by one allocation, which an object-identity check would miss. All eight device strings must also agree. The existing `validate_qwen4_tp1_measurements` then compares every shape, dtype, byte count, category set, and total with the checkpoint-derived footprint. A missing attribute, disabled PLE tensor, changed list structure, shared storage, dtype override, padding change, or byte mismatch fails closed.

`require_device` is deliberately limited to `None` or `'npu'`. With its default `None`, the report records the observed device but uses scope `POOL_OBJECT_TENSORS_MEASURED_TP1_PROFILE_VALIDATED_DEVICE_ONLY_RECORDED`; CPU/fake-object tests cannot label that result as device-qualified. Passing `require_device='npu'` requires every recorded device type to be `npu` and changes the scope to `LIVE_POOL_TENSORS_MEASURED_TP1_PROFILE_AND_REQUIRED_DEVICE_VALIDATED`. A qualified binding must use the latter form and persist the exact device strings and storage identities.

The minimal later binding seam is immediately after `self.init_memory_pool()` has returned in the model-runner initialization path. Calling before that point cannot work because the current framework overlay invokes `prefill_model_ready` and `ple_model_ready` before pool initialization. The binder should derive the footprint from its already verified config/profile, call the helper once with `require_device='npu'`, persist the returned eight measurements and source hashes in the device receipt, and reject startup on any mismatch. This helper is not yet installed in that seam and therefore makes no deployed or device-measured claim.

### F8 host/CPU contract

- Recompute all four table rows from `ContextGeometry`; reject unsupported context, non-page-aligned capacity, overflow, budget underflow, and non-BF16 current-consumer geometry.
- Record both checkpoint-derived TP1 planning bytes and observed tensor shapes. A gate fails if synthetic fixture totals are labeled as runtime budgets.
- For every page at first/last boundaries, bind distinct QSA_KV/QSA_INDEX identities and reject aliasing or an allocation-epoch mismatch.
- Recompute checkpoint-derived TP1 GDN and PLE byte formulas and compare them with the later device tensor shapes; do not substitute the current test placeholders of 4,096 and 64 bytes.

### F9 FP8 record and consumer equivalence, CPU first

- Deterministic round-trip for zero, finite random, maximum finite, and tail pages; byte-for-byte record stability including metadata and SHA-256.
- Corruption, NaN/Inf, invalid scale, wrong layer/shape, wrong request, stale epoch, duplicate completion, and load-after-release all fail closed without changing a published mapping.
- Index selection IDs are exactly equal to the BF16 baseline because the index remains BF16.
- For every QSA layer, compare baseline `sparse_gqa(q, full_bf16_kv, selected)` with compact staged BF16 output. The no-quantization staging case must be bitwise equal. The FP8 route must be finite and meet predeclared `max_abs`, relative-L2, and cosine thresholds on both adversarial and captured real page tensors. Set the numeric thresholds from a frozen calibration receipt before device qualification; do not choose them after observing holdout results.
- Peak host bytes equal the sum of live PageRecords plus a fixed transfer allowance; peak NPU staging bytes are no greater than the formula from selected-token bound, layer shape, and configured concurrent transfers. No per-token log or unbounded future exists.

### F10 allocator/lifecycle integration, synthetic NPU before long context

- With a small physical pool, cross at least three spill/reuse cycles. A freed page must be reassigned to a different logical page while the old page still reloads correctly from its host record.
- Capture `req_to_token`, logical page table, allocator free-page set, and physical-page generation at every transition. Assert no two live logical pages own one `(physical_page,generation)` and no returned page remains reachable through the Qwen4 consumer.
- Exercise page boundary decode, multi-page chunked prefill, QSA selection spanning resident and spilled pages, duplicate selected tokens, and the partial tail.
- Snapshot all 36 GDN states, the one PLE state/history, and the BF16 QSA index before and after spill/reload. GDN/PLE pointer identity and hashes must remain unchanged by the transfer itself; QSA index selection must remain exact.
- Inject export, quantize, host-write, digest, allocation, reload, dequantize, and completion-callback failures. Each must leave one authoritative copy and bounded credits. Abort/request end drains or invalidates callbacks, frees every physical/staging/host extent exactly once, and performs no migration.
- Prefix cache, speculation, disaggregation, graph capture, and memory saver remain off, matching the profile guard in `model_runner_kv_cache_mixin.py:353-368`.

### Device qualification and context ladder

After synthetic integration passes and the shared device is available, run 4K first with the unmodified BF16 consumer as control, then BF16 compact staging, then host FP8/BF16 staging. Evidence must include the TP1 pool shapes and data pointers, allocator/page-table transitions, transfer byte counts and latency, stage peak bytes, exact selected IDs, output error metrics, GDN/PLE/index snapshots, allocator cleanup, and generated-token agreement against the frozen control.

Proceed sequentially through 16K, 64K, and 128K only when the preceding target meets its frozen memory, quality, and lifecycle gates. A successful allocation without selected-token equivalence and page-reuse evidence is not a context-cache qualification. Native NPU FP8 remains `UNQUALIFIED` until a separate scale-aware Qwen4 write/read/consumer test proves that route; the reference host-FP8/BF16-stage result must not be relabeled as native FP8.

## Concrete blockers to avoid

1. **Free after generic CPU copy:** the hybrid wrapper does not implement that copy API, the generic implementation is CUDA-specific, and physical mappings/index entries would become stale.
2. **Global `kv_cache_dtype=fp8`:** it also changes QSA index storage and currently supplies unit scales; storage allocation is not numerical qualification.
3. **Reload all history before every QSA layer:** this recreates the full BF16 working set and provides no NPU capacity win.
4. **Reuse disaggregation retraction:** it clears recurrent request ownership and omits QSA-index/PLE identity.
5. **Keep freed positive slots in `req_to_token`:** allocator reuse turns them into aliases. Sentinels require tier-aware model, backend-metadata, allocation-tail, and cleanup paths before any page is returned.
6. **Spill GDN or PLE with token pages:** their state advances per request and must survive as the same live slot throughout generation.

The implementation boundary is therefore narrow but real: logical BF16 index ownership, a Qwen4 request-local page table/release path, and selected-page BF16 staging at `Qwen4ExpAttentionDecoderLayer.self_attention`. Everything else can retain the current reference numerical primitives and recurrent-state ownership.

## Independent PLE binding cross-check

This check is separate from the context-tier design. No scheduler/API mismatch was found in the current `runtime/ple_binding.py`:

- The real `Req` stores `mamba_pool_idx` as a scalar tensor (`managers/schedule_batch.py:642-645`); `HybridReqToTokenPool.alloc()` assigns `mid[0]` (`mem_cache/memory_pool.py:540-558`). The binding's `int(request.mamba_pool_idx)` converts that scalar before `PLEConsumerRuntime.begin`, whose state-slot contract requires a positive exact int.
- The model class is exactly `Qwen4ReferencePLE`; the checkpoint creates it only at zero-based layer 1 and computes 16 n-gram heads and width 160 (`models/qwen4_exp.py:466-487,554-557`). Its bound table is `CheckpointPLETable`, whose `reader` is `ShardedBF16Rows` (`:85-107`). The binding's one-layer, layer-id, head-count, reader-class, and width checks match those concrete objects.
- The installed scheduler order is `dynamic scheduler_before -> framework_before/ple_before -> run_batch -> process_batch_result -> framework_after/ple_after -> dynamic scheduler_after` (`managers/scheduler.py:1128-1149`). Thus the PLE lease starts after the placement generation is published, consumes the same `batch.input_ids` as the model, drains after model execution and result processing, and closes before the next dynamic boundary. Finished-request cleanup may already have freed the recurrent slot by `ple_after`, but that function only retires host credit by request id after the model consumer has completed; it does not dereference the freed slot.
- The model-runner overlay calls `ple_model_ready` before `init_memory_pool` (`tools/install_framework_overlay.py:47-55`) so its persistent ring allocation and transient-workspace reservation exist before KV profiling. The patched prepare seam retains the synchronous table fallback and the original model remains the sole history/state writer.

Two evidence limitations remain. The latest preserved v32 full-model source snapshot contains `ple_async.py` and `ple_rows.py` but no `ple_binding.py` or `ple_runtime.py`, so v32 is not evidence that this binding ran. Also, `ple_gate.py:7-38` deliberately source-locks only the component files and declares scope `COMPONENT_ONLY_NOT_MODEL_OR_OVERLAP_PERFORMANCE`; current binding tests cover default-off/rejection and source patching, not a real scheduler/model lifecycle. An F4 claim therefore requires a fresh overlay manifest containing the binding/runtime hashes and a real serial-model receipt showing bind, prefill/decode drain, finish/abort credit retirement, unchanged GDN/PLE ownership, and clean request end. This is a qualification gap, not a discovered mismatch in the reviewed API wiring.
