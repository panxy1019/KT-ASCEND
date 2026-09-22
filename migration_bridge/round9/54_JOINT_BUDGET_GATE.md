# F6 joint host-budget interaction gate

Date: 2026-09-21  
Status: bounded physical15 component gate v3 PASS; full-model interaction pending.

## Executed evidence

`evidence/physical15/20260921-ple-v3/result.json` records
`PASS_JOINT_BUDGET_SYNTHETIC_CONTENTION`; the owned container exited0.
The prerequisite PLE v2 receipt SHA-256 is
`b753270612776c26b15d2c00e9bb9817bf9d86ca8bcf7e12c183e455ffa2e551`.
All nine checks and six exact ledger snapshots passed. Measured extra allocation
was20,154,368bytes. The two synthetic destination buffers matched the independent
expected BF16 payload and the fixed digest below. This establishes only the
stated transport/credit scope; real full-model joint operation remains pending.

## Purpose and claim boundary

The F6 gate qualifies one narrow interaction: an original-row asynchronous PLE transfer and a synthetic Q32-sized expert transport share one `HostWorkBudget` without exceeding configured credits, blocking required migration behind optional prefetch, releasing storage early, or retaining request credits after completion.

The gate does not run expert arithmetic, load a model, publish KTEP maps, assess placement quality, measure useful overlap, or qualify full-model inference. Its receipt scope is exactly:

`SYNTHETIC_Q32_TRANSPORT_CREDITS_NOT_EXPERT_MATH_OR_MODEL`

Runtime joint mode must remain unavailable until a fresh receipt passes `require_joint_budget_device_gate` and is bound by path and SHA-256. Independent PLE and Dynamic component receipts are prerequisites, not substitutes for this interaction receipt.

## Bound implementation

The receipt binds exact hashes for:

- `host_budget.py`
- `budgeted_migration.py`
- `ple_async.py`

It also binds the verified original model/source identity, Torch and Torch-NPU versions, the standalone PLE gate SHA-256, the probe source through launcher admission, and physical device 15.

The runtime verifier is `runtime/joint_budget_gate.py`:

```python
require_joint_budget_device_gate(
    path,
    expected_sha256,
    source_identity=verified_source_identity,
    source_sha256=exact_three_file_hashes,
    ple_gate_sha256=standalone_ple_gate_sha256,
    torch_version=str(torch.__version__),
    torch_npu_version=str(torch_npu.__version__),
)
```

Any missing or additional receipt field, source mismatch, changed receipt bytes, incomplete check set, non-exact lifecycle snapshot, missing lifecycle stage, changed geometry, or malformed transport case fails admission.

## Exact budget

The Q32 checkpoint expert payload is derived from the audited shape:

`2,560 hidden * 640 intermediate * 3 projections * 2 BF16 bytes = 9,830,400 bytes`

The full Q32 migration-volume credit is:

`32 experts * 9,830,400 = 314,572,800 H2D bytes`

The PLE slot payload is:

`512 rows * 160 width * 2 BF16 bytes = 163,840 bytes`

The gate requires the same configured limits used by joint runtime binding:

| Resource | Capacity | Reserved from optional PLE prefetch |
|---|---:|---:|
| Host | 40,304,640 B | 39,321,600 B |
| H2D | 314,900,480 B | 314,572,800 B |
| Pinned | 327,680 B | 0 B |

The host reserve covers the existing adapter's four-expert bounded host staging. The H2D reserve covers the complete Q32 migration volume. Two persistent pinned PLE slots consume the pinned capacity. Two full PLE payload copies plus spare headroom account for the nonreserved host capacity.

The physical probe migrates two synthetic expert payloads, or 19,660,800 actual H2D bytes. This keeps the component probe small while exercising the real acquisition, in-flight, H2D, synchronization, completion, and release path. The verifier separately requires the exact full-Q32 budget arithmetic above; the two-expert execution cannot be relabeled as a 32-expert migration qualification.

## Physical probe sequence

`tests/joint_budget_device_probe.py` performs the following bounded sequence:

1. Verify frozen launcher admission, the three interaction source hashes, the probe hash, standalone PLE receipt SHA, original checkpoint identity, logical-device-0 to physical15 mapping, a single visible NPU, and the sole `/dev/davinci15` node.
2. Cap the Torch-NPU allocator to one percent and record its baseline.
3. Create the exact configured `HostWorkBudget` and a real two-slot `AsyncPLERing` over `ShardedBF16Rows` from the original checkpoint.
4. Submit and lease 512 original PLE rows on a real NPU consumer stream. Keep their host, H2D, pinned, event, and request credits live.
5. Submit a bounded optional PLE prefetch into the second slot and keep its credits live.
6. Run `BudgetedMigration.copy_experts` for two 9,830,400-byte synthetic expert payloads through a real Torch-NPU adapter. The adapter performs actual CPU-to-NPU copies and captures a budget snapshot while the migration allocation is in flight.
7. Require migration progress while both PLE allocations exist, verify the current PLE bytes exactly match the original reader, cancel and drain the optional prefetch, retire the shared request, close the ring, and require every allocation, request cursor, in-flight marker, and used byte counter to return to zero.
8. Validate the generated receipt with the same runtime gate before atomically publishing it.

The synthetic adapter writes deterministic BF16 payloads only to prove transport: slot 0 contains 4,915,200 copies of BF16 `8.0`, and slot 1 contains the same count of BF16 `12.0`. After synchronization, the probe copies the destination to CPU, compares both slots with an independently constructed expected tensor, and requires the fixed byte SHA-256 `8eca02bdec774408ad743e89af338ac0f5a65e38532d9b25e5d1ce7ef8b28896`. This is evidence of a completed synthetic copy, not evidence of correct checkpoint expert weights or inference.

The recorded NPU allocation peak must be at least 19,988,480 bytes: the 19,660,800-byte two-expert destination plus the two persistent 163,840-byte PLE device slots. The one-GiB verifier ceiling and the probe's one-percent allocator cap bound unexpected workspace growth.

## Mandatory checks and snapshots

Every receipt must contain all true checks:

- `physical_mapping`
- `ple_current_overlap`
- `migration_h2d_exact`
- `required_progress_with_prefetch`
- `prefetch_cancel_reclaim`
- `no_capacity_exceeded`
- `request_retirement`
- `zero_credits_after_close`
- `full_q32_budget_math`

It must contain exactly one snapshot for each stage:

- `ple_current_live`
- `prefetch_live`
- `migration_live`
- `after_migration`
- `after_cancel`
- `final`

Each snapshot records used host/H2D/pinned bytes, allocation count, active request count, and in-flight count. The verifier requires the exact derived ledger below, in addition to enforcing the configured capacities. This prevents a receipt made from plausible but fabricated below-capacity counters from passing.

| Stage | Host B | H2D B | Pinned B | Allocations | Active requests | In flight |
|---|---:|---:|---:|---:|---:|---:|
| `ple_current_live` | 327,680 | 163,840 | 327,680 | 3 | 3 | 1 |
| `prefetch_live` | 337,920 | 168,960 | 327,680 | 4 | 3 | 2 |
| `migration_live` | 39,659,520 | 19,829,760 | 327,680 | 5 | 3 | 3 |
| `after_migration` | 337,920 | 168,960 | 327,680 | 4 | 3 | 2 |
| `after_cancel` | 0 | 0 | 327,680 | 2 | 3 | 0 |
| `final` | 0 | 0 | 0 | 0 | 0 | 0 |

The three active request cursors before final cleanup are the joint request and the two persistent ring-slot reservation owners. `after_cancel` proves transient current, prefetch, and migration credits were reclaimed; ring close then releases the persistent slots and retires all three cursors.

## Receipt schema

The exact top-level fields are:

```text
schema, status, scope, physical_device,
source_identity, source_sha256, ple_gate_sha256,
torch_version, torch_npu_version,
expert_bytes, resident_capacity, migration_probe_experts,
configured_budget, checks, snapshots, case, peak_extra_bytes
```

The fixed status is `PASS_JOINT_BUDGET_SYNTHETIC_CONTENTION`. The case binds 512 PLE rows, width 160, the 163,840-byte PLE payload, two migrated expert payloads, 19,660,800 actual migration H2D bytes, 314,572,800 full-Q32 budget bytes, and SHA-256 values for the copied PLE and synthetic migration tensors.

## CPU verification

`tests/test_joint_budget_gate.py` constructs the complete schema and verifies admission, then independently mutates source hashes, check results, reserved capacity, full-Q32 arithmetic, an over-capacity snapshot, a below-capacity but non-exact snapshot, final lifecycle state, receipt hash, and the prerequisite PLE SHA. Every mutation must fail closed.

CPU tests alone do not produce a physical15 claim; the separately archived v3
device receipt above supplies the narrow component evidence.
