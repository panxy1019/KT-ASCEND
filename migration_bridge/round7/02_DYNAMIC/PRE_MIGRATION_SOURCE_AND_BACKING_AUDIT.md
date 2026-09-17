# Pre-migration source and CPU backing audit

Evidence state: **CODE_INSPECTED / LOCAL_VERIFIED**, not A3 Dynamic qualification.
Static release-v6 continues unchanged. Future probes must use a separate tool
release and must wait for healthy static serving to finish and reap.

## Independent source identity

Destination weights equal to full-context weights alone would miss a
self-consistent wrong logical-expert source. The diagnostic Dynamic auditor now
compares first/last migrated slots per layer against the frozen BF16 Hugging Face
checkpoint selected by logical layer/expert ID. It checks every copied slot
against the full-context source as well. Logical tensor orientation may transpose
but may not change projection dimensions or dtype. Readbacks remain absent from
observer-OFF timed performance.

## CPU backing-pool defect candidate

Actual deployed C64 source:
`/workspace/c64-dev/source/kt-kernel/operators/llamafile/moe.hpp`,
`LLAMA_MOE_TP::load_weights`, lines404–409: multi-subpool strided weight copy
returns early when `config.should_skip_expert(logical_id)` is true.
`operators/common.hpp`, lines255–257 defines that predicate to include experts
currently marked NPU-owned. The actual wrapper installs its pinned initial
ownership mask before loading and calls the normal load task. Dynamic later
updates this same mask in place.

Therefore an initially NPU-owned expert later evicted to CPU could have a
reserved but uninitialized CPU weight slice. This is a **defect candidate**, not
yet a reproduced A3 root cause. Uniform/Frequency static operation never changes
ownership and its current results must not be downgraded on this hypothesis.

The audited official pinned implementation has the same load-time skip at the
LLAMAFILE seam; official Dynamic presence is not proof for this C64/F32 profile.
Primary reference: [pinned official LLAMAFILE source](https://github.com/kvcache-ai/ktransformers/blob/0b954fe4ca2f9e79519da66156e6a3f92b686232/kt-kernel/operators/llamafile/moe.hpp).

## Minimal behavioral regression prepared

`tools/cpu_backing_probe.py` loads the ordinary CPU wrapper under its frozen
initial profile, then uses the official in-place wrapper mask update to evict
two predetermined initially resident experts while preserving slot budget.
For each evicted expert it compares three real historical operands with the
canonical BF16-visible CPU reference, requires relative-L2<=0.01, finite output
and ten-repeat exactness. Early/middle/late layers should be separate processes
to preserve the CPUInfer singleton topology.

This diagnostic does not alter a production mask, CMMLU selection, weights or
threshold. Execute the original binary first and preserve its result. Only a
reproduced failure justifies a focused load-time backing-pool fix; a repaired
binary must rerun the same regression, local operators and necessary static
compact tests before full-model Dynamic qualification. Do not overwrite the
binary used by the healthy current static driver.
