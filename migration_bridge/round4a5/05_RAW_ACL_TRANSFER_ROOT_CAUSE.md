# Round 4A.5: Raw ACL transfer root-cause attribution

Status: `ROOT_CAUSE_ATTRIBUTED; FIX_QUALIFICATION_PENDING`

## Scope

This report attributes the P2 same-path nondeterminism to the Ascend bridge
between NPU tensors and the CPU-expert backend.  It is not a claim that CANN
or LLAMAFILE independently has a numerical defect.  The tested production
path used raw `acl.rt.memcpy` calls with tensor data pointers; that bypasses
PyTorch/torch_npu ownership of stream dependencies and transfer lifetime.

## Fixed test envelope

Every run below retained the frozen P2 GGUF, placement, 16 CPUInfer workers,
one NUMA subpool, `max_running_requests=1`, seed 0, 64 generated tokens, and
`SGLANG_KT_HYBRID_NO_CPU_STREAM=1`.  The sequential-control setting is
important: the earlier failure survives when SGLang's separate CPU stream is
disabled.

## Controls and result

| Run root | Transfer variant | Prompt | Repeats | Exact | Unique output hashes |
|---|---|---|---:|---|---:|
| `p2-input-clone` | raw ACL D2H/H2D; clone D2H results before CPU task | `v_en_01` | 10 | no | 7 |
| `p2-torch-transfers` | PyTorch-managed synchronous D2H/H2D | `v_en_01` | 10 | yes | 1 |
| `p2-torch-transfers-struct` | PyTorch-managed synchronous D2H/H2D | `v_struct_01` | 10 | yes | 1 |

The `p2-input-clone` result rejects the narrower hypothesis that merely
reusing the raw ACL D2H destination buffer corrupts CPU computation.  The two
positive runs use the same allocations, CPU task, routing, arithmetic, and
request history; they replace only the transfer API:

```text
raw ACL:        acl.rt.memcpy(host_ptr, device_ptr, ...)
diagnostic A/B: destination.copy_(source, non_blocking=False)
```

The framework path lets torch_npu establish the NPU stream dependencies and
keep the source/destination tensor lifetimes coherent.  Both historical
failure classes converge to exactly one response hash, whereas the raw path
was already known to fail under the identical sequential control.

## Attribution boundary

The evidence supports this precise conclusion:

> KTransformers' raw ACL pointer-transfer integration is the P2 trigger.  It
> does not participate in the framework's NPU stream/lifetime model, allowing
> a nondeterministic transfer visibility/order outcome during full serving
> lifecycle execution.

The evidence does **not** prove an internal CANN library defect, nor does it
make the old CPU worker-count, scratch allocator, or SGLang staging observations
production fixes.  Those A/Bs were useful localizers but are not needed to
explain the two successful transfer controls.

## Required follow-through

1. Promote framework-managed D2H/H2D to the normal Ascend CPU-expert path.
2. Remove the `KT_DEBUG_NPU_TORCH_TRANSFERS` environment gate after code review.
3. Re-run all frozen P2 acceptance checks, preserving evidence manifests.
4. Keep P3 `NOT_RUN` until P2 passes its full acceptance criteria.

The persistent A3 artifacts are:

- `/home/admin/kt-artifacts/round4a5/p2-input-clone/repeats-v_en_01.json`
- `/home/admin/kt-artifacts/round4a5/p2-torch-transfers/repeats-v_en_01.json`
- `/home/admin/kt-artifacts/round4a5/p2-torch-transfers-struct/repeats-v_struct_01.json`
