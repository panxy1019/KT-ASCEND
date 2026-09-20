# Round9 real KTEP local merge gate

2026-09-20, .3 physical15. PASS for a bounded layer0 module fixture. This is
not a scheduler, full-layer shared-expert, full-model or performance result.

The probe uses the actual KTEP constructor, weight creation, post-load path,
runner creation and `apply` coordinator. NPU computation uses f02's actual
`UnquantizedFusedMoEMethod`: native routing, grouped matmul, SwiGLU and finalize.
CPU computation uses the new Ascend-enabled LLAMAFILE C64 wrapper and the
original BF16 bank qualified in27. TP rank is supplied as0 in this isolated
fixture; no distributed process group or model scheduler is claimed.

64 NPU experts use mask logical_id%8==0. All64 gate/up and down tensors are
loaded from the fixed original checkpoint, transposed and cast through native
ND->NZ post-load processing, then converted back and checked bit-for-bit
against their original tensors. Every route is assigned to exactly one owner
by KTEP's actual mask/remap. Forward tests exercise selected experts including
logical0,504 and CPU511; they do not numerically exercise every resident expert.

T1/10/32 × mixed/resident-only/repeated-CPU routes =9 cases,3 repeats each:

- Every repeat hash is identical; all partial and final outputs are finite.
- Maximum CPU partial relativeL2:0.0001138027073466219.
- Maximum NPU partial relativeL2:0.00043756733066402376.
- Maximum merged relativeL2:0.000697444484103471.
- Disabled-owner partials are exactly zero.
- Captured CPU and NPU partials are FP32. Actual merged output is exactly
  `(cpu_partial+npu_partial).bfloat16()`; no earlier partial-sum BF16 cast.
- Native NPU peak allocation1,048,747,520 bytes; container exited0.

The independent CPU reference reads original selected safetensor weights,
materializes BF16-visible gate/up/activation and route weights, uses FP32
CPU down partials and BF16-visible NPU down output, then performs a single
BF16 merge cast. The threshold remains1%; no tolerance changes were used.

Resources: CPUquota72, whole-node affinity480–639, two32-worker pools on
NUMA6/7, RAM/swap20GiB, pids512, SYS_NICE, no network, physical15 only.
No training container or global NPU configuration was changed.

Evidence: `evidence/physical15/20260920-ktep-bank/`; executable fixture
`tests/ktep_bank_probe.py`. Source/binary hashes identify the exact implementation.
Full model still requires real loader execution, shared expert/gate exactly-once
verification, request-bound PLE/GDN/QSA integration, final GR, resource receipts,
and the separately approved full-model correctness/quality tests.
