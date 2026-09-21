# Context reference component gate — 2026-09-21

Status: CPU v5 and bounded physical15 QSA component v4 PASS.
Full-model KV quant reference, HiCache and long-context qualification remain open.

Follow-up CPU v6 passes300 tests and installer validation. The added
`context_reference_gate.py` binds the actual v4 receipt to exact source identity,
device, software stack, all12 numerical cases, bounded peak and final ownership
drain. Mutation tests reject changed scope/device/source, incomplete cases,
failed exact output, nonfinite error metrics and live remaining leases. This
validator is groundwork for later model admission, not itself model integration.

## Implemented path

`context_staging.py` accepts one query's bounded selected logical token IDs and
reads identity-bound BF16 host records, FP8 reference records, or synchronous
resident CPU pages. It preserves order, duplicates and padding while remapping
the selected IDs to compact BF16 rows for the existing `sparse_gqa` consumer.
Request, allocation epoch, page, layer, causal boundary, finite values and shape
are checked before returning the completed result. No physical allocator is used.

An initial implementation retained every decoded selected page and per-row clones
until the final stack. Sparse selection could therefore keep many more rows than
the final compact capacity. Review changed it to plan destinations first,
allocate final K/V once, and consume one source page at a time. A weak-reference
test checks that each source page is released before the next callback.

The final2051-row TP1 K/V allocation is bounded by4,200,448bytes, plus one decoded
BF16 page of8,192bytes. Caller-owned records, Python metadata and FP8 codec
temporaries are separate; this formula is not a measured whole-process peak.

## Actual CPU and NPU evidence

`evidence/physical15/20260921-framework-cpu-v5/tests.log` records298 passing tests,
including actual Torch BF16 compact-QSA and FP8 codec cases. The associated
container inspection shows no mapped NPU devices. The same snapshot successfully
installs the overlay against the preserved real SGLang baseline.

`evidence/physical15/20260921-ple-v4/result.json` records
`PASS_CONTEXT_REFERENCE_QSA_COMPONENT`. The namespace uses the guarded component
launcher; `probe_kind=context` and the exact source hashes distinguish this from
a PLE test. The container exited0 on exclusive physical15.

For each of the12 real QSA layer identities, the probe creates eight synthetic
BF16 KV token rows on NPU, exports both four-token pages across all12 layers to
canonical host records, then stages the selected rows back to NPU. Selection
includes reordering, a duplicate and padding. Output is exactly equal to the
original full BF16 NPU cache oracle.

The FP8 path quantizes the same source pages on CPU, validates and decodes their
records, and passes compact BF16 to the NPU QSA consumer. Its output is exactly
equal to a full decoded-page NPU oracle. Relative-L2 against original BF16 is
recorded separately; maximum across the12 fixtures is0.03765484318137169, below
the explicit synthetic-fixture bound0.10. This is not population quality evidence.

Measured extra NPU allocation is355328bytes for this small fixture. Invalidation
with a live host lease waits for release; the retired lease is rejected, and
final host bytes/pages/leases/reloads are zero. `allocator_free_calls=0` and
`native_fp8_qualified=false` are explicit in the receipt.

## F8 measurement binding

`--context-target` explicitly selects4096/16384/65536/131072 pool sizing and writes
a config-hash-bound probe profile into launch evidence. The requested server
capacity is target+4 and context length is target+2. A hook after
`init_memory_pool()` measures eight actual root tensors, requires NPU storage,
checks ownership/nonaliasing and compares every shape/dtype/byte count with the
checkpoint-derived TP1 footprint. The flag is absent by default.

This binding passed CPU contract and real-source installer checks. It has not yet
measured a full-model allocation because current NUMA admission is blocked.
Even a future pool-measurement pass will mean allocation geometry only, not a
16k/64k/128k generation qualification.

## Remaining work

The model still needs request-logical QSA index ownership, a physical/logical page
table, safe allocator release and backend metadata handling, and actual consumer
integration before any HiCache claim. Prefix sharing remains disabled. Streaming
must have its own weight/context/PLE lifecycle and full-model gate. Ordinary
chunked prefill does not satisfy that requirement.
