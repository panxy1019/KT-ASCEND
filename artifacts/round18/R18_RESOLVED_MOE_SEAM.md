# Round18 resolved native MoE seam

## Frozen identities

- Image: `quay.io/ascend/vllm-ascend:qwen3.8-next-a3-openeuler`
- Image ID: `sha256:7e3ab7e56a54259d3ecec91073346f9ca8b2a4c1f58e15826e3ed6ff67036f00`
- vLLM commit: `568afb3a13806beb53bb2e6bd518269357b237c0`
- vLLM-Ascend commit: `550ab66e7feb89f6e2ba49394f6ffb153189ac2b`
- Frozen Ascend `fused_moe.py` SHA-256:
  `2d03cf6963f4e7edd529751383c779c407560011760c86478cf0598661d47f7c`
- Frozen `prepare_finalize.py` SHA-256:
  `c467d0ed0a0d41525aa90008aafbfd26bdbb718b0e29db7de2bed9abe09cabdd`
- Qwen revision: `de4b8e4d43b917e7706784d8bb445c9af86a3540`

## Actual call and ownership boundaries

1. The original Ascend selector produces the complete logical-512 Top-10 in
   `AscendUnquantizedFusedMoEMethod.apply`.
2. The same method passes those IDs, their original weights, the physical
   expert tensors, and `expert_map` to the frozen communication/token-dispatch
   implementation.
3. `AscendMoERunner.no_shared_forward_impl` owns prepare, the routed method
   call and finalize.  The runner's outer forward retains shared expert,
   routed scaling/transform and residual semantics.
4. The upstream factory already separates `routed_experts_cls` from
   `runner_cls`.  The vLLM-Ascend platform patch defaults the latter to the
   concrete `AscendMoERunner` and preserves an explicitly supplied owner.
5. The narrow Round18 seam therefore injects only the already-qualified Q32
   owner.  After the native runner is constructed with its original
   logical512/EP1 config, the execution map is rebound to the Q32 owner's
   512-entry `global logical → physical slot/-1` tensor.

## Questions answered before device execution

| Question | Frozen-source answer |
|---|---|
| Where is full Top-K produced? | `AscendUnquantizedFusedMoEMethod.apply` via the original `select_experts`. |
| Who applies route weights? | The native token dispatch/MLP/combine path receives the original `topk_weights`; the bridge must not multiply them again. |
| Who owns shared expert/finalize? | The unmodified `AscendMoERunner` and its native communication path. |
| Can the owner be Q32 while router stays logical512? | Yes at construction/loading level; Round16 V1/V2 proved exactly 32 physical rows and a selective loader. |
| Does frozen kernel skip map value `-1` in EP1? | Yes. `native-partial-v004` proved all-nonresident produces the exact zero partial and mixed cases match the independent resident-only reference. |
| Is a custom full runner allowed? | No.  The installer fails if `runner_cls` or `runner_args` is supplied. |

## P1 device conclusion

The first device call used the exact original selector once and invoked the
frozen `AscendUnquantizedFusedMoEMethod` with logical `num_experts=512`, Q32
NZ29 weights and the owner's native NPU map.  It tests all-resident,
all-nonresident, 1/9, 5/5, 9/1, multi-token and tied-Top-K cases against an
independent checkpoint BF16 reference.

The run passed.  The frozen native kernel accepts the Q32 global-to-slot map,
skips `-1` nonresident occurrences and does not require a compact adapter.
The largest relative L2 was 0.00264708; 20 reuse calls were stable.  This
opens P2 without changing the native runner.

## P2 seam

P2 wraps only the routed owner quant method.  During the existing native
selector call it submits the same logical Top-K tensors to the qualified KT
layer wrapper.  The native method then computes the resident partial exactly
as in P1.  The adapter drains the CPU partial and adds it once to
`FusedExpertsResult.routed_out`; the untouched runner subsequently performs
its original finalize/shared/scaling/residual work.  A process lock restricts
the selector observation to the frozen TP1, single-request scope, and every
work identity includes layer and monotonically increasing call generation.
