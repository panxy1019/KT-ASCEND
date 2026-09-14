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

## Gate

```text
R6_LOCAL_STATIC = PASS
R6_LOCAL_RUNTIME = NOT_RUN_NO_TORCH
R6_A3_REQUIRED = YES
```
