# B Round 2B Summary

## Repository

- round2a final sha: `540ccbc28b1fa9327b90dc86c244e0df707409d8`
- round2b base: `540ccbc28b1fa9327b90dc86c244e0df707409d8`
- round2b final: this documentation commit; use `git rev-parse HEAD`
- branch: `feature/kt-ascend-runtime-plane`

## A3 Runtime

- OS: openEuler 24.03 LTS-SP1 host; Ubuntu 22.04.5 disposable container
- CANN: 9.0.0
- driver: 26.0.rc1, inner `V100R001C10SPC001B257`
- torch: 2.9.0+cpu
- torch_npu: 2.9.0.post2

## Build

- Ascend-enabled: PASS on aarch64; public CANN Runtime only
- CPU-only regression: PASS; Ascend OFF and no ACL dynamic dependency
- CANN headers: `/usr/local/Ascend/cann-9.0.0/aarch64-linux/include/acl/acl_rt.h`
- CANN library: `/usr/local/Ascend/cann-9.0.0/aarch64-linux/lib64/libacl_rt.so`

## Device Stream

- python stream type: `torch_npu.npu.streams.Stream`
- native handle: public non-zero integer `npu_stream`, A3 verified
- C++ cast: `uintptr_t` → borrowed `aclrtStream`
- ownership: torch_npu/Python; CPUInfer does not create/destroy it
- status: PASS

## Host Callback

- API: `aclrtLaunchHostFunc(aclrtStream, aclrtHostFunc, void *)`
- single callback: PASS
- 10k stress: PASS, exact 10,000 callbacks
- multi-stream: PASS, exact 1,000 callbacks on each of two streams
- callback thread: CANN-managed; no identity/affinity assumption
- status: PASS

## CPUInfer

- submit callback: PASS
- sync callback: PASS
- overlap: PASS; conservative lower bound 27.032 ms
- 1000 cycles: PASS

## Pinned Memory

- PyTorch pin_memory: supported and verified
- allocator: PyTorch/torch_npu pinned allocator
- aclrtMallocHost fallback: BYPASSED because not required
- D2H: BF16/int64/float32 exact PASS
- H2D: BF16 exact PASS

## Runtime Pipeline

- D2H → CPUInfer → H2D: PASS using real Round 2A LLAMAFILE MoE
- numerical: max abs `4.768e-7`, mean abs `1.863e-9`, relative L2 `2.297e-6`
- ordering: later NPU verification and terminal callback markers PASS
- RSS: final 1,000-cycle delta 5 MiB, limit 16 MiB

## Lifecycle

- stream ×100: PASS
- CPUInfer ×20: PASS
- callback args: explicit RAII ownership and host-thread error channel
- leaks: fixed `SyncArgs` and MoE task-parameter leaks; stress thresholds PASS

## P0 Rules

- callback launches device work: NO
- callback synchronizes device: NO
- torch_npu private C++ ABI: NO

## CPU Regression

- Round2A core matrix: **21 passed in 5.67s**, 1,000-forward RSS delta 0

## Exit Gate

```text
CPU_EXPERT_PLANE = A3_VERIFIED_READY
ASCEND_RUNTIME_PLANE = A3_VERIFIED_READY
```

Round 2C prerequisites are satisfied. This does not itself authorize or implement Round 2C.

## Remaining blockers

No Round 2B blocker. The Round 2A generic-aarch64 ISA portability risk remains outside this runtime-plane scope. NUMA memory-policy enforcement is still not claimed in the non-privileged container.

## Commits

1. `d8da19a` device-neutral callback abstraction
2. `495b10c` Ascend CANN vendor adapter
3. `2087b98` public native stream handle bridge
4. `6ed3b4f` host callback ordering tests
5. `b120c84` CPUInfer callback/overlap tests
6. `54bc1ec` pinned transfer tests
7. `d2d8be0` callback lifetime fix and lifecycle/error tests
8. `d07604c` full runtime pipeline tests
9. this documentation commit
