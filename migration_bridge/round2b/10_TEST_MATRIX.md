# Round 2B Test Matrix

## Final A3 Ascend matrix

The seven focused modules collected 24 tests and completed with **23 passed, 1 skipped in 28.70s**. The skip is the CUDA-only compatibility-handle case in an Ascend-only container; all Ascend cases executed.

| Gate | Result |
|---|---:|
| Ascend vendor adapter compile/error semantics | PASS |
| public torch_npu native stream handle | PASS |
| single callback ordering | PASS |
| 10,000 callbacks | PASS |
| two-stream ordering | PASS |
| CPUInfer submit/sync callbacks | PASS |
| CPU/NPU overlap | PASS |
| PyTorch pinned allocator | PASS |
| BF16/int64/float32 D2H | PASS |
| BF16 H2D | PASS |
| D2H → real CPUInfer MoE → H2D | PASS |
| 1,000 full cycles | PASS |
| stream lifecycle ×100 | PASS |
| CPUInfer lifecycle ×20 | PASS |
| callback state/error paths | PASS |
| callback contains device work/sync | PASS (source audit) |
| Ascend-enabled build/import | PASS |
| fresh Ascend-OFF build/no ACL dependency | PASS |
| Round 2A core CPU matrix | PASS, 21/21 |

## Exit-scope checks

- model code changed: NO
- Hybrid MoE implemented: NO
- SGLang model integration changed: NO
- torch_npu private C++ ABI used: NO
- callback launches or synchronizes device work: NO
- host/other cluster workloads modified: NO

Authoritative combined logs: `logs/round2b/ascend_full_matrix.log` and `logs/round2b/cpu-regression.log` in the retained A3 audit copy.
