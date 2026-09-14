# Round 6 Local Tests and Build

Evidence state: `LOCAL_VERIFIED` for static checks; runtime tests remain
`NOT_RUN_LOCAL` because the workstation environment does not provide Torch or
an Ascend runtime.

## Production change

SGLang child commit:

```text
eaa58ccea4ee372b8b207dd16f398eafe3da37ff
fix(round6): guard deferred block terminals
```

The change adds `resolve_layer_max_deferred()` at the server-argument to
`KTConfig` boundary.  A layer may defer only when its immediately following
physical layer also has a KT CPU wrapper.  This preserves contiguous Deferred
blocks and converts all terminals—including the historical sparse P1/P2/P3
placements—to defer zero.

## Added tests

- Child topology tests cover a contiguous producer, a block terminal, the
  model-final layer, sparse Round 5-style placement, negative values, invalid
  layer indices and invalid mask rank.
- Parent selection tests cover the immediate/deferred route partition,
  flattened-batch expert protection, zero/full protection bounds and
  equal-score repeat determinism.
- A standard-library AST/source guard checks that the producer/consumer entry
  points exist, topology resolution is wired into `KTConfig`, the two-slot
  contract is present and raw `acl.rt.memcpy` is absent from the parent bridge.

## Executed locally

```text
python3 migration_bridge/round6/tools/verify_deferred_source_contract.py
  ROUND6_DEFERRED_SOURCE_CONTRACT=PASS

python3 -m py_compile <all newly changed Python files>
  PASS

git diff --check
  PASS
```

## Deferred to A3

- Torch unit tests and SGLang Ascend test module.
- Actual immediate/deferred selection telemetry.
- Empty-deferred pending behavior.
- CPUInfer queue ordering, two-slot generation and incremental exactly-once.
- Model build, serving and every numerical/quality/performance gate.

## A3 instrumentation build and final-state rebuild

The opt-in Deferred scheduling trace and empty-task suppression were rebuilt
inside the disposable Round 6 A3 container with the frozen CPUInfer/Ascend
flags.  The final wheel build reported SHA256
`5a6e7abb87c8a47f962a498d146cdbdd91f151bf456b52315fbbf117b699ff24`.
The parent Deferred-selection suite passed `5/5` with the installed wheel.
The detailed mechanism result is recorded in
`07_D1_DEFERRED_MECHANISM.md`.

After the one-shot pending-state consumption fix, A3 was fast-forwarded to
parent `b6815196d8bfd531f810ba34372999dc77911b9c` (SGLang child
`eaa58ccea4ee372b8b207dd16f398eafe3da37ff`). The disposable container
rebuilt and installed `kt-kernel` with the same CPUInfer/Ascend flags. The
final-state wheel build reported SHA256
`508e519d0d727269712e2c1596621a94ca96113719dd6525f7d5c8f73207fba8`;
the installed extension SHA256 was
`3fbb95c00c5906b905e8fd689346f4e78f86714fddb6372fa0e481e0da7ce43f`.
The final build log SHA256 is
`d3436e7825415fdd0bbe8dd2a94f3678d386b9ec588e5dda9fe564187243f05a`.

On that installed final-state wheel:

```text
ROUND6_DEFERRED_SOURCE_CONTRACT = PASS
parent Deferred selection tests = 6/6 PASS
SGLang Ascend alignment contract tests = 7/7 PASS
```

These results are `A3_VERIFIED` targeted tests, not full D1 qualification.
Raw logs are retained under `/home/admin/kt-artifacts/round6-deferred/a3-final-tests/`
on A3.

## Gate

```text
R6_LOCAL_STATIC = PASS
R6_LOCAL_RUNTIME = NOT_RUN_NO_TORCH
R6_A3_REQUIRED = YES
```
