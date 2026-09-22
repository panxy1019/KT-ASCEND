# F2/F3 — bounded layerwise transport and integration

Date:2026-09-21. This stage is IN PROGRESS, not framework completion.

## v32 full-model result and external-resource interruption

The fresh keyword-compatible binding completes both HOLDOUT requests through
all48 layers. Results are `391` and `Seawater contains salt.`; both responses
are finite and their token IDs equal the Q32 Hybrid reference v29. Maximum
output-logprob differences are0.004972559865564108 and0.02407521102577448.
This records short differential output evidence, not arithmetic equivalence:
the private layerwise path reads original BF16 expert weights while Hybrid
uses MXFP4 for nonresident routed experts.

At2026-09-21T05:04:41UTC the exclusive guard detected new physical15 PID3425392,
belonging to the unrelated `mzy-train` container, and stopped only our container.
The last owned memory observation was26546MiB and the container was not OOM-killed.
The requested active-abort/reuse driver then found the container already stopped;
that gate is NOT_RUN, not PASS or a model correctness failure. Short Hybrid
fallback, multichunk lifecycle and fresh-process restart remain outstanding.

Raw evidence for failed v31 and interrupted v32 is retained in
`evidence/physical15/20260920-full-model-v31` and`20260920-full-model-v32`.
Do not resume NPU work without fresh resource admission or a newly confirmed
shared-training envelope. CPU development continues independently.

## Implemented path

An independently owned32-slot BF16 NZ29 expert bank receives original checkpoint
experts through the existing qualified projection copier. Contiguous token rows
are grouped so each token retains its complete Top10 routes while each group
uses no more than32distinct experts. Group outputs remain FP32 until the final
BF16 model-path cast. A complete layer span synchronizes before buffer reuse.
Resident Q32/Q64 tensors, masks and mappings are never used as scratch storage.

The shadow is a new module with an audited scalar allowlist, no inherited hooks,
registered/unregistered tensors or mutable caches. A copy-only facade exposes
the existing copier without placement publication or controller binding.
Request identity, placement epoch and buffer allocation epoch remain distinct.

The real scheduler coordinator admits serial plain EXTEND batches up to32tokens.
The threshold uses total prompt length. Each chunk must execute all48layers;
telemetry covers contiguous input spans across chunks. Decode and short prompts
retain the hybrid path. Prefix sharing, speculative/overlap schedules and the
old full-expert prefill mode are rejected. Schema1 remains unchanged; schema2
adds a default-OFF explicit prefill contract bound to a real component receipt.

## Review fixes and negative gates

- Bare unquantized NPU output originally lacked the qualified FP32-contribution
  flag. Set it explicitly and cast once after complete route accumulation.
- Expert-count limits do not bound the number of token rows. Added a receipt-bound
  maximum32rows and reject larger chunks before allocation/execution.
- Final allocation delta misses transient weight format conversion. Measure the
  constructor peak including conversion, runner and private maps; failures poison
  the backend and clean partial ownership.
- Failed synchronization must not release in-flight slots or hide the primary
  error. Explicit BROKEN_RELOAD_REQUIRED retains unsafe ownership until reload.
- Shallow copying the resident module could alias future hooks/caches. Replaced it
  with explicit private construction. Sentinel copier maps are hidden behind a
  copy-only interface and cannot become a published placement.
- Coordinator telemetry/event errors originally left open phases. Added phase-aware
  abort cleanup, between-chunk cancellation even during idle/decode, immutable
  request/span/epoch identity, and stateful failure tests.

## Real component gate

Probev1 never started device execution: the monitor correctly rejected its newly
introduced container namespace. Added only `kt-round9-layerwise-v[0-9]+` to the
owned-container allowlist, preserving rejection of unrelated/training names.
The RED ownership test then passed with all14monitor tests.

Fresh probev2: PASS_LAYERWISE_BF16_TRANSPORT. Frozen original-model/native/runtime
identity; original-source address samples for all48layers; numerical execution
explicitly only fixture layer0. Six cases cover1/10/32tokens and one/multiple
route groups. Relative-L2 range0.00203–0.00299, each below0.01. All output finite
BF16; original template weights/maps unchanged; copied shadow weights exact;
unused22slots unchanged and numerically ignored; in-flight/public abort reuse pass.

Measured extra allocation peaks:

| Interval | Bytes |
|---|---:|
| Private-bank constructor including transient conversion |524289536|
| Execution relative to persistent template plus shadow |616490496|
| Probe allocator ceiling |2631841873|

Evidence: `evidence/physical15/20260921-layerwise-v1` and`v2`.
The strict gate rechecks the source SHA, receipt SHA, identities, geometry,
measurements, six numerical cases, ownership and lifecycle; Q32 migration receipts
cannot substitute for this transport gate.

## Full-model integration work

The private bank is created before KV/GDN pool sizing. Its persistent weights
reduce available memory; the remaining projection/workspace budget is explicitly
deducted by the Qwen KV profiler before GDN/token-capacity calculations. This is
independent of the unrelated upstream MXFP4 reservation field, whose observed
consumer is a different model's profiler.

Trial31 first hit the unchanged per-NUMA80GiB free admission. Reclaimed only own
inactive checkpoint/BF16/Q8/MXFP4 file caches with POSIX_FADV_DONTNEED; no global
drop, file mutation or threshold relaxation. File metadata and absence of live
Round9 bank mappings were checked. Fresh identical admission passed.

Trial31 then loaded the complete model and created its private bank, but failed
before the first expert dispatch: real FusedMoE calls `apply(layer=...,
dispatch_output=...)`, while the wrapper used incompatible argument names.
Added a direct real-keyword contract regression; changed the wrapper to preserve
that call signature. Trial31 remains FAIL. Trial32 uses a fresh overlay with only
the binding and its regression test changed from trial31's frozen source.

Full-model layerwise engagement, short content/differential checks, activeabort,
multi-chunk behavior and fallback/OFF regression remain pending at this revision.
No performance, product quality or long-context qualification is claimed.
