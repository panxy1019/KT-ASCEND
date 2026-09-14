# Overlapped CPU/NPU Hybrid MoE

## Schedule

`HybridMoECoordinator.forward_overlapped` (`kt-kernel/python/hybrid_moe.py:294-317`) performs:

```text
NPU D2H copies to pinned host buffers
host callback -> CPUInfer submit
concurrent NPU routed-expert compute
host callback -> CPUInfer queue sync
pinned host output H2D
explicit merge-boundary synchronization
clone shared device buffer and add CPU + NPU contributions
```

The explicit synchronization is outside the host callback. It closes the required merge dependency while preserving CPU/NPU compute overlap.

## Buffer ownership

| Buffer | Owner | Writer | Reader | Lifetime | Location | Pinned |
|---|---|---|---|---|---|---:|
| `input_host` | `KExpertsCPUBuffer` | NPU D2H copy | LLAMAFILE CPU task | static buffer cache / completed before reuse | host | yes |
| `ids_host` | `KExpertsCPUBuffer` | NPU D2H copy | LLAMAFILE CPU task | same | host | yes |
| `weights_host` | `KExpertsCPUBuffer` | NPU D2H copy | LLAMAFILE CPU task | same | host | yes |
| `output_host` | `KExpertsCPUBuffer` | LLAMAFILE CPU task | H2D copy | same | host | yes |
| shared device output | `KExpertsCPUBuffer` | H2D copy | coordinator clone | synchronized before clone | NPU | no |
| `npu_output` | NPU provider/result | NPU operators | coordinator merge | result lifetime | NPU | no |
| `merged_output` | result | coordinator add | caller | result lifetime | NPU | no |

## Correctness and race found during final matrix

Sequential and overlapped CPU, NPU, and merged tensors are required to be bitwise equal in the overlap tests for qlen 1/8/32. The first combined 43-test launch exposed one `qlen=32` mismatch after smaller-buffer cases. Classification: `H2D / BUFFER_LIFETIME / MERGE`. The fix in commit `b8ba787` synchronizes the queued H2D before cloning the shared static device buffer, then synchronizes the merged result. Two consecutive full matrices subsequently completed with `43 passed` each.

## Overlap proof, not a performance claim

Final A3 interval measurement:

```text
CPU interval             2.121210 ms
NPU interval             2.359660 ms
wall interval            3.134158 ms
conservative overlap     1.346712 ms
```

The positive lower bound proves real temporal intersection. It is not a throughput, latency, or production performance claim.
