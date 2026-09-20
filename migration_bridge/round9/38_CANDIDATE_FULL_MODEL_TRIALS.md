# Candidate full-model functional/resource trials

User authorized exclusive physical15 on110.123.0.3 until they request sharing.
Preserved baseline remains unchanged; no performance tests or timer restart.
No candidate inherits quality qualification or the previous partial256 reference.
Native candidate buildv2 fixes Q8_0 layout and low-bit FP32 TP partial merging.
All entries require fresh128physicalworker/NUMA2,3,6,7/136CPU/640GiB/RAM admission,
onlyphysical15 mapping, live own-process guard and verified48-bank/source identity.

| Trial | Candidate | Current result |
|---|---|---|
| v10 | C128/BF16/static | PASS functional/resource checks below; stopped normally |
| v11 | C128/BF16/request-boundary Dynamic | FAIL CLOSED: actual full-model NPU layout was not qualified ND2; no migration admitted |
| later | C128/Q8_0/static and Dynamic | Banks+local gates verified; full trials pending |
| later | C128/MXFP4/static and Dynamic | All48 banks independently verified; full trials pending |

## v10 measured evidence

[evidence/physical15/20260920-full-model-v10/trial-summary.json](evidence/physical15/20260920-full-model-v10/trial-summary.json)
binds the five completed driver reports, effective profile, code payload,
resource admission, server log and final monitor log.

- 48 layers loaded from newly rehashed original BF16 banks.
- First real HTTP prefill/decode PASS.
- System8: four prompts×two repeats, exact static repeat trajectories PASS.
- Four fixed known-answer canaries PASS.
- Actual3840input+256generated=4096token resource request PASS.
- Active abort then identical clean request reuse PASS.
- Actual128 single-pinned scheduler threads on128 distinct CPUs.
- Peak observed board42624MiB; owned39856MiB; cgroup309354479616bytes
  before controlled shutdown (about288.1GiB), memory.failcnt0.
- No resource guard stop; only owned v10 stopped after all checks completed.

This is PASS_CANDIDATE_SYSTEM_RESOURCE_ONLY. It is not complete quality256,
restart equivalence, performance, or deployment qualification.

## Exit telemetry correction

Original monitor reread/proc/PID/cgroup each sample. During controlled shutdown
that process moved to root cgroup while Docker still said running, causing three
post-stop samples to report host memory as owned memory. Raw logs and the
incorrect trial-summary-pre-cgroup-fix.json are retained. Corrected summary scopes
peaks to timestamps before controlled-stop.json, yielding309354479616bytes;
it does not discard any inference-time sample. NPU counters were unaffected.

Future guard pins the container-ID-bound memory cgroup and validates its limit
against Docker admission, records its path, and never substitutes root cgroup.
12 targeted monitor tests pass, including root-cgroup and failed-Docker-snapshot
negative cases. v11 was already frozen before this change; apply corrected
monitor from the following trial onward and retain pre-shutdown scope for v11.

## Remaining interpretation boundaries

Dynamic test events record original logical IDs and resulting slots for all
changed layers. A successful request-boundary system run is not a same-placement
fresh-static full-model oracle or a fresh256 quality score. These gates remain
explicitly pending until separately executed. No TTFT/TPS ranking is produced.

## v11 layout mismatch and corrective gate

The full SGLang NPU backend enables torch_npu.npu.config.allow_internal_format,
whereas the earlier bounded expert fixture left it disabled. Thus v11 reached
real scheduler binding but rejected the resident tensor layout before migrating
any expert. Container exited137 after SGLang child failure; DockerOOMKilled=false.
This is a failed integration trial, not a full-model Dynamic PASS. Existing ND2
results remain correctly scoped to ND2. New bounded expert-v10 explicitly enables
internal format and asserts actual NZ29, then requires exact original expert
identity after each of3 migrations and real KTEP merge before model retry.
The adapter default still only allows ND2; NZ29 is an explicit gated probe option.

## NZ29 repair verified —23:07UTC

Expert-v10 stopped before doing useful probe work: the first cgroup hardening
incorrectly chose hybrid0::/ instead of explicit v1memory. Exact host membership
reproduced RED in a new test, then fixed selection to prioritize v1memory.
Expert-v11 then reproduced the real NZ slice-copy failure at step0/slot0/expert511.

Fix: for explicitly admitted NZ29, call index_copy_ on the actual resident tensor,
never copy_ on a slice that can be an independent ND materialization. Check the
resident's format is still29 after the write. Existing ND2/CPU behavior remains.
Expert-v12BF16, v13Q8_0, v14MXFP4 all pass real NZ29 identity for all64 resident
experts after3 migrations, format invariance, pinned mask pointer and nine KTEP
cases×3 repeats. BF16 probe ownedHBM peak1960MiB. No untested NZ allow-list bypass:
fullmodel Dynamic requires the matching bank, source, workers and format gate,
exact generation/move sequence and all9 finite/repeat numeric rows. Launcher also
checks candidate adapter/dynamic/source modules exactly match the device probe.

All48 MXFP4 banks finished CPU conversion; static and Dynamic CPU receipts each
rehash all bank bytes and verify actual type, geometry, source and file set.
31 ARM CPU tests pass; P2 gate binding supplement passes targeted local tests.
Actual corrected cgroup guard fixes the Docker-ID-bound path and exits normally
on the successful NZ probes. Fullmodel-v12BF16+C128+Dynamic now loading using
candidate-sglang-v6 and fresh exclusive resource admission. Result still pending.

## v12 controlled diagnostic stop and bounded layer staging

v12 actually committed generation1 across48layers/424experts and generation2
across8layers/8experts. The first transaction exceeded4minutes; sampled Python
stack stayed in real NZ migration and source read counters advanced. CPU cgroup
throttling was negligible. It was stopped deliberately before full driver
completion and is INCOMPLETE_CONTROLLED_DIAGNOSTIC_STOP, not a model PASS.

Each single-expert NZ index_copy_ can process the whole resident projection.
The correction stages one layer projection in ND (at most400MiB) by streaming
original expert payloads under the existing64MiB host limit. It then mutates the
resident NZ tensor once for that projection and synchronizes before releasing
the temporary buffer. Publication still happens only after all layer copies.
A new real device test reverses all64expert slots and restores them, checking
every original BF16 value before normal3migration+9KTEP checks. The full-model
gate requires that new evidence; old single-expert gates cannot qualify it.
State-machine batch/failure tests were observed RED then GREEN. Expert-v15 is
currently running; no corrected device/full-model PASS yet.

## 23:18UTC: batch gates passed, full combinations started

BF16 expert-v15, Q8_0 expert-v16 and MXFP4 expert-v17 all exited0 with
PASS_REAL_KTEP_LOCAL_MERGE, including64expert reverse/restore and the unchanged
3migration/9case matrix. ARM CPU host-v23 passed29tests; host-v22 failed only
because torch backend auto-loading lacked the NPU environment, then host-v23
explicitly disabled device backend auto-loading for these CPU-only tests.

Independent review found noP1. The receipt field max_device_staging_bytes is an
explicit tensor bound, not an allocator/workspace measurement. BF16-v15 observed
ownedHBM peak2364MiB versus1960MiB in previous single-expert NZ probe:404MiB
additional observed process footprint. Full-model actual peak/guard remains the
final resource evidence; no allocation estimate is labeled full-model PASS.

One-shot sequential matrix (not a timer) began: full-v13 BF16/Dynamic, full-v14
MXFP4/static, full-v15 MXFP4/Dynamic, full-v16 Q8_0/static, full-v17 Q8_0/Dynamic.
Each starts under fresh admission, runs five drivers, stops only its owned
container and finalizes immutable evidence before the next. Any failure stops
sequence for diagnosis. Candidate-sglang-v7 contains the qualified batch adapter.

## v13 allocator-cap failure and exclusive-only correction

v13 completed firstHTTP and two real Dynamic commits, then index_copy_ failed
requesting402MiB. PyTorch reported38.38GiB active,39.13GiB reserved,39.50GiB
allowed and21.72GiB free device memory. DockerOOMKilled=false. This is an
allocator resource gate failure, not a successful trial or hostOOM. Matrixv1
stopped before any later profile. Single-layer observed deltas did not predict
full-model allocator behavior, exactly why full-model gate is separate.

Root cause: copied baseline model constructor retained40448MiB allocator cap
from shared-card execution. The isolated installer now patches only candidate
constructor to43008MiB(42GiB) and requires exclusive mode. Candidate server checks
fresh receipt: training0, allocator43008MiB, process guard45056MiB(44GiB), model
admission46080MiB(45GiB), reserve8192MiB plus live baseline. Monitor accepts the
higher44GiB only for exclusive mode; shared upper bound remains41472MiB and
absolute board guard56832MiB stays active. Baseline source is unmodified.

22 relevant tests PASS including explicit rejection of the larger budget under
shared mode. Same adapter/native device gates apply; candidate-sglang-v8 is a new
isolated payload. Fresh matrixv2 uses full-v18 BF16Dynamic, v19 MXstatic,
v20 MXDynamic, v21 Q8static and v22 Q8Dynamic. Full resource results pending.
