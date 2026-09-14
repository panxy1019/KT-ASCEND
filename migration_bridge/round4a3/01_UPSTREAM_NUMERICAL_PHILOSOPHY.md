# Upstream Numerical Philosophy and Local Evidence

Evidence state: **CODE_INSPECTED** plus prior **A3_VERIFIED** round evidence.

## Exactness remains mandatory for system semantics

The Round 2C overlap regression requires sequential and overlapped CPU/NPU
partials and their merge to be bitwise equal. That gate found a real static
buffer lifetime/H2D ordering defect and remained exact after the fix. Round 4A
likewise demonstrated a byte-identical CPU-not-hit control. These are execution
semantics, not cross-backend floating-point comparisons, so no numerical
tolerance is applicable.

The exact N0 set remains router IDs/weights, ownership, logical-to-physical
mapping, physical expert count, routed scaling, shared expert ownership,
CPU/NPU disjointness, buffer lifetime, and stream ordering.

## Cross-backend arithmetic is bounded, not assumed bitwise identical

Round 4A.1 captured identical hidden states, source weights, router IDs, and
routing weights and then localized the residual to deterministic expert GEMM
differences under different reduction/blocking behavior. The same expert may be
bitwise equal for one input and non-exact for another. BF16-visible boundaries
and FP32 Hybrid accumulation were implementation defects and were fixed before
acceptance work began.

Round 4A.2 then compared pinned LLAMAFILE, OpenBLAS, BLIS, ATLAS, and ACL paths.
OpenBLAS closed the `v_struct_03` tie but opened a different `v_en_01` tie; both
production trajectories remained 42/45. Therefore selecting a backend because
one known greedy tie flips in the desired direction is not a corpus-wide
correctness solution.

## Contract consequence

Round 4A.3 separates:

1. exact system semantics and same-path determinism;
2. bounded expert and matched-history logit error;
3. 100% exact top-1 agreement in stable-margin positions;
4. membership in a bounded baseline tie set at near ties; and
5. paired downstream quality evidence.

Only Q may derive the numerical envelope. H tests generalization without
refitting. A stable-region token flip, an out-of-set near-tie choice, an
envelope overflow, nondeterminism, or a significant quality regression remains
a hard failure.
