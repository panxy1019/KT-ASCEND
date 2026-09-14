# Physical/Logical Mapping Matrix

## Frozen direction

```text
physical_to_logical_map[physical_id] = global_logical_id
```

Example `[2,0,3,1]` means physical slots 0,1,2,3 contain logical experts 2,0,3,1 respectively. It does **not** mean logical-to-physical.

The LLAMAFILE loader validates the permutation at `kt-kernel/python/utils/llamafile.py:171-201`. C++ copies each physical slot into the mapped logical resident slot at `kt-kernel/operators/llamafile/moe.hpp:194-258`. The NPU fixture uses the inverse only as an implementation step to reorder physical source tensors into global-logical indexing (`hybrid_moe.py:151-160`).

## A3 matrix

| Mapping | qlen | Router weights | Result |
|---|---|---|---:|
| `[0,1,2,3]` | 1, 8, 32 | standard and edge matrix | PASS |
| `[2,0,3,1]` | 1, 8, 32 | standard and edge matrix | PASS |
| `[3,2,1,0]` | 1, 8, 32 | standard and edge matrix | PASS |
| `[1,3,0,2]` | 1, 8, 32 | standard and edge matrix | PASS |

The focused routing/mapping launch completed with **16 passed in 8.97s** and worst max absolute error `0.000244140625`.

Invalid duplicate, short, or out-of-range mappings fail before weight load (`test_hybrid_placement.py:65-68`).
