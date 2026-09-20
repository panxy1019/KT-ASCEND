# Development-first proposal: C128, CPU low-bit, request-boundary Dynamic

Date: 2026-09-20. Status: REVIEWABLE_DESIGN; candidate implementation NOT_STARTED.
Latest user direction: prioritize framework development, defer device testing to
an idle window tomorrow, and exclude performance tests. No schedule is created;
Round7/Round9 automations remain PAUSED. Target remains .3 physical15.

## Saved baseline and paused quality work

Q64/BF16-storage/F32 CPU/C64 synchronous PLE reference passes the local/module,
full-load, compact32, cancellation/reuse, restart, observerOFF and actual4096
resource gates. CMMLU first128 complete:106 correct,0 invalid. Final256 reference
is INCOMPLETE; second128 not started at user direction. Do not use106/128 as a
qualified256 score or compare a candidate256 score against it. Exact manifest,
driver and v6 runtime identity must be preserved when completing the reference.

v6 container stopped after scoring; docker State Running=false, OOMKilled=false,
ExitCode137 reflects bounded docker-stop teardown, not a clean server exit proof.
Monitor observed docker-top failure during intentional teardown and attempted
STOP_OWNED; do not label this as a training/resource failure. Own process peak
39868MiB, board48726MiB. Final read-only physical15 HBM8969MiB, only training
PID4069720 remains on this device. No owned Round9 container/guard remains active.

## Recommended architecture and development order

Prefer integrated candidate paths with separate switches and fail-closed capability
checks, keeping the frozen baseline immutable. Alternatives are empty interface
stubs (fast but hide native gaps), or finishing each candidate end-to-end before
starting the next (slower to establish all three seams). The recommended middle
path implements configuration, conversion/state logic, real runtime seams and
host-side validation now; native/NPU eligibility is granted only after real tests.

### 1. Candidate profile and capability contract

Add a separate candidate profile/schema and validation path rather than editing
`runtime/profiles/q64_bf16_c64_sync_physical15.json` or weakening its admission.
Represent cpu topology, CPU routed storage and placement policy independently.
Keep NPU TP1/Q64, originalBF16 resident weights, synchronous PLE, batch1/inflight1,
Graph/radix/spec/MTP/DeferredOFF. Unsupported native combinations must fail with
a reason, never silently fall back to another precision/pool count.

Current model seam: `sglang_overlay/.../models/qwen4_exp.py:_prepare_kt_runtime`
requires C64/two pools/NUMA6,7/DynamicOFF. Add a separately selected candidate
validator instead of deleting these baseline checks. Candidate runtime hashes
and gate evidence must be distinct from the baseline.

### 2. C128 topology and partitioning

Use four32-worker NUMA subpools and intermediate slices[0:160],[160:320],
[320:480],[480:640], F32 partials and deterministic merge before finalBF16 cast.
Do not count SMT siblings as physical workers. The current visible480–639 CPU
set contains only80 physical cores, so changing cpuinfer64 to128 is invalid.
Candidate admission must discover four suitable NUMA nodes,128 distinct workers
plus auxiliary budget (initially8, total136), physical-core identities, cpuset,
quota and memory locality. Node numbers stay unassigned until fresh resource
admission; old .22 NUMA IDs are not an A3 allocation.

Reuse `kt-kernel/python/experts.py`, `experts_base.py`, and
`utils/llamafile.py` multi-pool interfaces. Verify Python and native partition
construction agree. Reuse one complete CPU backing bank; no four complete
copies. Before any C128 device test require topology, partition completeness,
no-overlap, first-touch intent, merge ownership and source identity checks.
Development may wire BF16 C128 first; final choice of C64/C128 is still compared
at the same selected precision after local correctness qualification.

### 3. CPU low-bit bank and native capability path

MXFP4 remains the preferred candidate; Q8_0 is the fallback. Retain original
BF16 source for exact NPU promotion; quantize only CPU routed experts.
A new source-bound converter needs explicit block scale/rounding/packing rules,
streamed expert/projection conversion, independent decode reference, metadata,
checksums and atomic no-clobber publication. Do not generate all48 banks before
small-sample format/native correctness passes. Complete512-expert backing is
required before enabling demotion for a whole model.

Concrete gaps found in current code:
- `tools/mxfp4_gguf/convert_mxfp4_gguf.py` repacks native DeepSeek-V4 MXFP4; it is
  not a BF16 Qwen post-training quantizer and cannot be reused unchanged.
- `python/utils/llamafile.py` permits tail-safe multi-pool640 only for F32/BF16;
  MXFP4/Q8_0 must not be enabled by merely deleting the guard.
- `operators/llamafile/moe.hpp:forward_many` has a256-alignment condition outside
  the F32/BF16 tail path. C64 slices320 and C128 slices160 expose this real native
  gap even though both formats use32-weight blocks.

Implement format-aware partition/capability metadata and targeted tail handling
only after native format/vec-dot audit. An unavailable ARM kernel returns explicit
UNSUPPORTED_NATIVE_KERNEL, while conversion/decode tests can still complete.
No all-pool F32 expansion and no false nativeFP4/BF16 ISA claim.

### 4. Bounded request-boundary Dynamic

Start with serial TP1/batch1 request boundaries, not live token-level mutation.
Sequence: collect route statistics -> select deterministic64 logical experts/layer
-> validate plan -> quiesce all relevant work -> stage bounded originalBF16
expert projections -> finish device copies -> publish consistent weights,
logical/physical maps, CPU/NPU ownership masks, backing identity and generation.
A request observes one coherent generation; no publication while a request or
state consumer is active. PLE/GDN/KV request cleanup remains mandatory.

The existing KTEP `_update_gpu_experts_from_batch` sources a temporary full GPU
context and updates mapping/masks in place. Do not enable that flag directly:
for512 experts it defeats Q64 memory bounds. Round7 auditor also embeds a
DeepSeek checkpoint path/64-expert assumptions; reuse its invariants, not its
model-specific implementation or qualification.

Implement a bounded original-source promotion provider, route-statistics policy,
plan/diff, generation state machine and publication seam. Preserve surviving
resident slots where possible, allocate only admitted staging, and record copied
bytes/changed IDs (diagnostics only, no performance test). If a copy/publication
fails after mutation, block new requests; resume old generation only after a
verified rollback, otherwise fail closed and require reload. Never assume metadata
rollback restores overwritten device weights.

CPU low-bit backing must cover512 experts/layer; NPU promotion always uses
originalBF16, not dequantizedCPU weights. For early Dynamic math tests use the
already qualified BF16/C64 backing to isolate placement from quantization.
A same-placement fresh-static oracle must begin from the same request history,
not compare states that have already diverged.

## Development deliverables before device tests

- Separate candidate configuration/capability validation and reproducible build
  commands; unchanged qualified baseline.
- C128 four-pool partition/topology plumbing and host-side rejection tests.
- MXFP4/Q8_0 converter/decoder integration with small deterministic fixtures;
  explicit status of native ARM kernels and320/160 grouped-prefill support.
- Dynamic planner, generation/publication state machine, bounded promotion seam,
  negative cases for active request, stale generation, duplicate IDs, partial
  transfer and rollback failure.
- One test entry point with independent stages and evidence manifests; no default
  performance stage and no automatic heavy conversion/model load.

Acceptance terminology: IMPLEMENTED_HOST_TESTED, NATIVE_BUILD_VERIFIED,
DEVICE_TEST_PENDING, UNSUPPORTED. None means candidate-model PASS. A framework
must either execute its real seam or reject with a specific capability gap;
empty placeholder callbacks are not implementation completion.

## Tomorrow's idle-window validation order

1. Fresh host/training/NPU/NUMA/resource admission; no standing reservation from
   today's15GiB user-attested training envelope or current idle snapshot.
2. Complete baseline remaining128 on the preserved v6 runtime and observerOFF
   profile; perform restart4 and exact artifact/raw audit to obtain full256.
3. C128/BF16 local native and partition/merge tests, then bounded system smoke.
4. MXFP4 same-quantized-weight local canonical tests on C64; Q8_0 fallback if
   unsupported or numerically invalid; test320/160 explicitly before combinations.
5. BF16/C64 Dynamic forced-trigger/non-trigger, original-weight identity,
   generation isolation, copy-failure/reload and same-placement static comparison.
6. Combine only independently passing candidates; compact/restart/cancel first,
   then frozen256 quality versus the complete baseline if time permits.

No TTFT/TPS/ABBA throughput ranking or performance adoption decision in this
batch. Do not promise the entire combination matrix in one idle window. Keep
capacity/system/quality qualification separate from later performance work.
No automatic timer restart; tomorrow's exact window has not been specified.
