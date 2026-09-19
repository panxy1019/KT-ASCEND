# Round9 — Heterogeneous Working-Set Runtime

Authorization: user mandate v1.0, 2026-09-19. This is an execution contract,
not evidence of model or hardware qualification. Read latest user instructions
first. Round8 remains the active heavy workload until its evidence is audited
and final state locally committed. No Qwen3.6 or Round8.5 branch.

## Current checkpoint

PREPARED_WAITING_ROUND8. Only small pinned model metadata was downloaded.
No Round9 weights, conversion, container, production patch or device test was
started. See `manifests/preparation.json` and `CURRENT_EXECUTION.md`.

## Ordered milestones and evidence

| Phase | Work | Exit evidence |
|---|---|---|
| M0 | Close Round8 Qwen3.5 Q64 engineering reference and Q32 static product baseline | Full code/runtime/artifact identities, system/quality/stability evidence, authorized performance outcome and limitations, reproducible handoff, local final commit |
| R9-A | Freeze sources; inspect architecture and resource feasibility | Full model revision, tokenizer/template/index identities, source compatibility matrix, tensor-header census, actual host/disk/cgroup/NPU admission |
| R9-B/M1 | Q64/F32 Static, synchronous host PLE, original QSA/GDN/GR | Early/middle/late real-operand tests; complete weight identity; 32 compact requests, repeat/restart/observer/state gates; frozen CMMLU256 reference |
| R9-C/M2 | One ARM CPU low-bit candidate, at most one technical fallback | Frozen format/reference contract; kernel vs decoded-weight reference; quality vs original Q64/F32; capacity and speed measured separately; adopt/reject decision |
| R9-D/M3 | PLE async, bounded cache and joint resource budget | Exact lookup and transfer, cancellation/reuse negatives, cold/warm tests; same-budget synchronous/async/coordinated comparison |
| R9-E/M4 | Q64 per-layer Frequency, then Dynamic | Independent DEV/HOLDOUT; freeze before held-out; generation/weight/mapping/mask consistency; triggered quality and net performance |
| R9-E/M5 | Independent nested Q32 static capacity point | Same precision/PLE baseline; system and CMMLU versus original Q64/F32; performance/capacity; Dynamic only after Q32 static passes |
| R9-F/M6 | 4096 → 16384 → 65536 → 131072 context | Actual input/output budget, causal/cache/selected-KV identity, known-answer long-range tasks, memory/time and highest verified tier |
| R9-G/M7 | Final audit and handoff | Raw-evidence hashes, per-profile state, defaults, limitations and rebuild instructions; local final commit |

M2 failure does not block independent PLE work on F32. An optimization may be
correct but not beneficial and remain OFF. An unimplemented low-bit path or
unverified context tier must remain explicitly incomplete, not hidden under M1.
Do not automatically run heavy Round9 tasks merely because a Round8 process exits.

## R9-A: source and admission contract

Use pinned metadata, then inspect real checkpoint headers before converters or
capacity claims. Record main model, PLE, MTP, vision, tokenizer/chat template,
generation config, license and index separately. Config BF16 is not proof that
every checkpoint tensor is BF16. Do not call dequantized weights original BF16.

Audit three code lines independently: actual validated local imported files and
native binaries; official KT pinned fork; architecture-supporting upstream
SGLang. Minimal semantic ports only, no wholesale upgrade. Inspect packed
gate/up axes, transpose, ND/NZ, shared/scaling owners and PLE layer numbering.
Resolve unknown mathematical semantics before full-model execution.

Refresh physical cores/NUMA, cpuset/cgroup, NPU logical/physical mapping, free
disk and loading peaks. Include checkpoint, F32 conversion, PLE, anonymous
copies, NUMA replicas, page cache and staging. Record swap baseline and deltas,
process VmSwap, faults and pressure; historical swap alone is not failure.
Never pin the whole expert pool or PLE table. If insufficient, record
BLOCKED_RESOURCE_ADMISSION; do not change precision/model/TP/host implicitly.

## R9-B: faithful static implementation

Text only, TP1, one in-flight request, explicitly fixed decode batch. Per main
MoE layer: 64 NPU residents out of 512; freeze masks before quality. CPU F32
expanded from exact source values; NPU BF16; complete CPU backing for all
logical experts. Dynamic/Deferred/Graph/MTP/speculative OFF.

C64 is the verified worker/subpool design, not TP64. Intermediate640 splitting
as320+320 is an unqualified candidate: verify alignment and FP32 partial merge.
Test early/middle/late layers after both GDN and QSA, first/last IDs, Top10,
repeated/mixed routes, ownership boundaries and real prefill/decode shapes.
Then stream-convert/audit all layers/experts without multiple450GiB copies.

Use same-operand NPU/production-CPU/canonical-CPU local references. Preserve
F32 local rel-L2<=0.01 under its defined operands/boundaries. Copies, layout
restoration and source-value expansion require exact identity.

PLE must fetch correct host table rows synchronously before optimization;
no zero/random/stale substitution or whole-table HBM placement. Verify hash/ID
history, BOS/EOS, shard/row order, request and chunk boundaries.
QSA must preserve indexer/block selection and causality; no replacement dense
or simpler attention labeled as original QSA. GDN must preserve prefill/decode
state and reset/cancel isolation. GR retains all four branches/read-write gates.
Slow correct kernels may be labeled REFERENCE_PATH, never falsely optimized.

Q64/F32 is an engineering model reference, not an independent All-NPU oracle.
Require module references, identity, canaries and template checks before using
its CMMLU score as reference. Claim PRODUCT_BASELINE_ESTABLISHED only within
the actually tested software/device/text/context scope.

## Low-bit contract

First modify only CPU routed-expert representation; keep NPU BF16, mask, PLE
and QSA/GDN fixed. Determine real ARM LLAMAFILE format support; x86 AVX2/AMX,
MXFP4 or NVIDIA examples are not Kunpeng support evidence. Freeze main format
and fallback reason before quality. Distinguish:

1. Kernel correctness against canonical decoding of the SAME quantized weights.
2. Representation quality against original Q64/F32 model.

Do not mechanically apply F32 0.01 to quantization versus original weights.
Keep complete low-bit CPU backing AND original BF16 source for future NPU
promotion, with manifests linked to the same original revision.
Outcomes include UNSUPPORTED_ON_ARM, BLOCKED_IMPLEMENTATION, QUALITY_REJECTED,
capacity-only benefit or adoption; never imply low-bit guarantees speedup.

## PLE async and working-set control

Preserve model/table/shard/row/layout identity and request/token/forward/history
identity. Track task, slot, completion event and consumer lease. Producer ready
before read, consumer complete before reuse; cancelled tasks cannot write into
a new request. Preserve gather ordering after deduplication.

Progress: CPU gather overlap → bounded pinned ring/framework copy/events →
bounded device cache. Decode prefetch uses only known token/history, not future
tokens. Report cold/hot hit, useful/wasted bytes and miss tail latency.
Current-token required data outranks speculative prefetch. Expert misses may
compute on CPU; PLE misses must fetch exact rows; QSA misses must supply exact
selected KV. Share resource budgets, not interchangeable mathematical semantics.
Compare independent bounded planes vs coordinated budgets under identical HBM,
precision, placement and workloads. No benefit permits a simpler default.

## Scheduler and Q32 contract

First Frequency is `frequency_per_layer_fixed_budget`:64 per layer. Official
global top-K allocation is a separate experiment, never silent substitution.
Freeze disjoint SCHED_DEV/HOLDOUT text/token IDs/buckets/generation before
results; no CMMLU answers or historical failure prompts for expert selection.
Use a small preregistered engineering screen, not unlimited candidate search.

Dynamic starts at quiescent request boundaries, then separately examines
prefill-end updates. Decisions may use only already-observed routing. Weights,
mapping, mask and CPU backing publish one generation. Partial copy failure
must fail-stop rather than serve half-updated slots. Test exact source identity,
tie behavior, repeat/restart and negative publication cases.
Fresh-static comparison requires SAME placement/input/state: don't ignore
different prior KV/GDN histories after in-request migration. Include equivalent
prefill-strategy Dynamic-OFF control and deduct full-context/migration costs.

Q32 is independent static qualification with nested masks and same original
quality reference, not inherited Q64 or Qwen3.5 qualification.

## Context and quality

k=1024. Verify actual4096/16384/65536/131072 effective context with output
reserve, no truncation or unauthorized RoPE scaling. QSA2048-token selection
does NOT establish bounded historical KV. Count indexer/KV/GDN metadata and
test selected-block identity. Host KV is a separately validated feature if
needed. Use fixed known-answer long-range retrieval/structured/code checks;
nonempty output and short CMMLU are insufficient long-context evidence.

Frozen CMMLU256 product regression uses observed delta>=-5pp against ONE
Qwen3.8 Q64/F32 reference: at most12 fewer correct,13 fewer fails. Do not reuse
DSV2/Qwen3.5 scores or compound5pp losses between optimizations. This repeatedly
used dataset is not independent statistical holdout. Freeze template/thinking,
greedy/max tokens/stop/parser/invalid handling before results. Preserve errors;
infrastructure retries keep original failure and fixed sample identity.
Dynamic reports trigger rate and adds frozen migration-triggered quality
coverage if ordinary CMMLU doesn't trigger. No answer-conditioned hotsets.
Legacy B_pair/logit/router metrics are diagnostics, not revived hard gates.

## Performance and final state

Fixed workloads/repetitions, same resources, no observers in timed paths;
formal performance isolated from other heavy tasks. Report startup/conversion,
TTFT/prefill/steady decode/E2E, CPU wait, PLE gather/copy, QSA/indexer costs and
real device timing, not Python return-time overlap. Memory includes logical
payload versus actual allocation/reservation/process HBM, host anon/file pages,
NUMA replicas and peaks. Don't inflate samples until a positive result appears.

Minimal ablations: F32+syncPLE+Static → CPUlowbit → asyncPLE → jointbudget →
Frequency/Dynamic → Q32 → context tiers. If low-bit rejected, explicitly use
F32 for later independent work. Capacity success and speedup are separate.

## Autonomy, safety and closure

Preserve failure → reproduce → inspect deployed source → official primary
sources → minimal patch → regression/negative control → repeat failed gate.
Fix ordinary software autonomously, but document unresolvable resource/device/
semantic blockers honestly. Do not weaken frozen inputs or semantics.

Current authorized execution host remains110.129.0.22; no new .16/.20/.3
deployment is implied by the word A3. Only owned workers may be stopped on
errors. No automatic test-farm expansion or unrelated task disruption.
No push, main merge, tags or Qwen3.6. Deferred defaultsOFF and is noncritical;
TP2/HCCL/Graph/MTP/speculative/multimodal/concurrency remain outside core scope.

Protect the existing dirty Qwen3.8 worktree. At M0 establish a new isolated
Round9 branch/worktree from the ACTUAL frozen baseline; don't borrow old
qualification or assume another branch is ready. Commit each semantic concern
and candidate before held-out use. Preserve raw artifact locations/hash/durable
copies, compact statistics by default, full tensors only for targeted failures.
Heartbeat remains active after M0; pause only after authorized scope is
truthfully closed, never call missing subgates complete. Notify meaningful
completion/failure/infrastructure need; stay quiet on unchanged runs.
