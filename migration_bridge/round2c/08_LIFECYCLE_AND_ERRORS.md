# Lifecycle, Stress, and Error Paths

## A3 lifecycle results

| Gate | Result |
|---|---:|
| 20 warm-up + 1,000 overlapped mixed forwards | PASS |
| deterministic first/final result | bitwise exact |
| RSS before | `1,857,671,168` bytes |
| RSS after | `1,857,671,168` bytes |
| RSS delta | `0` bytes (limit 16 MiB) |
| wrapper create/load/forward/destroy ×20 | PASS |
| alternating masks | `[F,T,F,T]` / `[T,F,T,F]`, PASS |
| NPU stream create/forward/destroy ×100 | PASS |
| zero native stream handle | fail-fast PASS |

Tests are in `test_ascend_hybrid_lifecycle.py:33-100`. The focused lifecycle launch completed with **4 passed in 15.38s**.

## Validation coverage

| Invalid state | Behavior |
|---|---|
| mask wrong length/rank | `ValueError` |
| mask not bool | `ValueError` |
| negative expert ID | `ValueError` |
| expert ID >= count | `ValueError` |
| duplicate/short/out-of-range permutation | `ValueError` |
| NaN routing weight | `ValueError` |
| unassigned expert | unrepresentable: exhaustive bool partition required |
| wrapper/provider placement mismatch | coordinator construction `ValueError` |
| deferred experts enabled | coordinator construction `ValueError` |
| zero stream handle | `ValueError` |

## Lifetime reasoning

Round 2B owns callback parameter RAII and its error channel. Round 2C retains the wrapper/provider/placement on the coordinator, uses the existing pinned buffer cache, waits for CPUInfer at the stream callback boundary, completes output H2D before cloning the shared device buffer, and returns independently owned contribution/result tensors. Host callbacks contain neither device launches nor device synchronization.
