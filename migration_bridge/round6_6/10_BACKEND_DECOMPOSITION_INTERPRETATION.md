# What the external-inspired reference lane established

## The comparison is now genuinely three-way

Production LLAMAFILE loads the unchanged frozen F32 GGUF. NPU uses actual Ascend
GMM with source BF16 values. Canonical CPU uses independent PyTorch/BLAS FP32
operations on exact-expanded source values. Pure FP32 and BF16-visible-boundary
references are separately named; neither is mislabeled an NPU reduction emulator.

B0 covers128 CPU expert instances,384 real-hidden decode cases and224 real
prefill cases,including11/16/64/128-row inputs. Complete coverage is not a claim
that every diagnostic expert was naturally routed at every input. Natural route
membership is retained explicitly.10-repeat checks and production/debug-stage
output parity cover all608 cases. Actual production intermediates were captured
by a default-off native test build,not inferred from the reference implementation.

Maximum production-vs-pure-FP32 rel-L2 is0.0044835914. The corresponding maximum
NPU-vs-pure-FP32 is0.0044823109. Production-vs-NPU maximum is0.0015245553.
These results do not show an order-of-magnitude LLAMAFILE-specific extra error.
The later52 naturally routed HD-tail cases also remain within original0.01.

## What is not a justified production fix

- Replacing CPU execution with NPU weight streaming would change the capacity /
  bandwidth architecture; it is not a correction to an identified CPU bug.
- Choosing a cast/reduction variant merely because it makes one observed token
  agree would be result-driven fitting. No such variant was installed.
- Moving troublesome experts to NPU after observing their failures would change
  the frozen placement. Numerical-aware residency remains future research.
- Increasing workers changes topology/performance,not the demonstrated cause.

No production arithmetic fix is made in this round because the new evidence did
not establish a violated operator invariant. Test-only reference seams and
capture instrumentation are the reusable implementation deliverables.

## Why full-model drift can still be larger

The operator result bounds a single same-input expert comparison. It is not a
Lipschitz bound on the complete multi-layer MoE. Hidden-state perturbations can
cross router rank-K boundaries; the resulting expert-set change is discrete.
The four-R32 historical experiments and new router traces expose this mechanism.
Being repeatable does not make a large full-model difference zero or universally
acceptable. It requires independent system/operator checks and task-quality
evidence under a new study identity.

## Evidence limits

The pre-frozen Round6.6 model layer reports distributions rather than asserting
a universal final-logit bound. HD6 exceeded the old B_pair at3 positions. Thus
the conclusion cannot be advertised as recovery of the Round6.5 old bound.
The engineering quality screen may qualify only the frozen four-R32 profile on
this model/runtime and workload. It cannot establish population equivalence,
all-MoE capacity readiness,Dynamic readiness or Deferred speedup.

External projects motivated the independent reference lane; their precision or
performance claims were not imported as our acceptance evidence. Sources and
their provenance limitations are in01_EXTERNAL_SOURCES.md.
