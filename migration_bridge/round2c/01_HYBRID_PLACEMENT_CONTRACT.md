# Fixed Hybrid Placement Contract

## Definition

For one layer with `N` global experts:

- `accelerator_mask[l] == True`: logical expert `l` is NPU-owned.
- `accelerator_mask[l] == False`: logical expert `l` is CPU-owned.
- `physical_to_logical_map[p] = l`: physical weight slot `p` maps to global logical ID `l`.
- Placement is immutable for the coordinator lifetime; no JSON loading config, deferred placement, cache-based reassignment, or dynamic migration is used.

`FixedExpertPlacement` is implemented at `kt-kernel/python/hybrid_moe.py:14-111`.

## Invariants

- Mask is a rank-1 CPU `torch.bool` tensor of exactly `num_experts` elements.
- Mapping is an exact permutation of `[0, num_experts)`.
- Every expert belongs to exactly one side because the bool mask and its complement define the partition:
  - `E_cpu ∩ E_npu = ∅`
  - `E_cpu ∪ E_npu = E_all`
- Routes are rank-2; ID/weight shapes match; IDs are in range; weights are finite.
- An “unassigned expert” cannot be represented by this contract. Wrong-sized/non-bool masks fail before a coordinator can exist.

## Partition behavior

- CPU route projection preserves CPU global IDs and replaces every NPU-owned ID with `-1` (`hybrid_moe.py:99-103`). C++ already treats `-1` and mask-true IDs as skipped.
- NPU projection preserves global IDs and zeros CPU-owned route weights before expert execution (`hybrid_moe.py:105-111`). Thus CPU-owned routes have exact zero NPU contribution.
- Coordinator construction cross-checks the CPU wrapper mask, provider placement/mapping, hidden size, and `max_deferred_experts_per_token == 0` (`hybrid_moe.py:214-239`).

## Verified fixture placement

Primary tests use four experts with:

```text
accelerator_mask = [False, True, False, True]
E_cpu = {0, 2}
E_npu = {1, 3}
```

`test_hybrid_placement.py:19-35` verifies exact partition, sentinel IDs, and NPU weight masking. Lines 52-81 cover wrong mask shape/dtype, invalid mapping, out-of-range/negative IDs, and NaN routing weights.
