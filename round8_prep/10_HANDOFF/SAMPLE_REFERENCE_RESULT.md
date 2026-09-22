# Layer0/E0 reference arithmetic — 2026-09-18

Pinned SGLang qwen3_5.py loader splits packed gate_up with `chunk(2, dim=-2)`,
loading first half to w1 and second to w3 (lines1718 onward in audited snapshot).
Used that convention with verified real Layer0/E0 sample, synthetic BF16 operands,
seed8675309, qlen1/8/64. Not captured model operands and not full MoE.

CPU lane: FP32 matmul, BF16 projection/multiply/output-visible boundaries.
NPU lane: torch BF16 matmul and SiLU/multiply, not grouped-MoE kernel.
CPU reference uses32 Torch threads only; this does not change or qualify the
production C64 two-subpool configuration.

| qlen | Output relative L2 (CPU reference vs NPU reference) | Finite | NPU 3-repeat |
|---:|---:|---|---|
| 1 | 0.0029681241 | true | exact |
| 8 | 0.0034417997 | true | exact |
| 64 | 0.0036105167 | true | exact |

These exploratory reference results are below0.01 but DO NOT close production
LLAMAFILE, grouped NPU MoE, C64, routing, shared, GDN or full-model gates.
Activation rounding differs between lanes; do not choose a production cast based
solely on proximity. Raw `sample-reference-v1.json` preserved locally and remotely.

Next inspect actual native BF16 backend capability before selecting it; current
local `python/utils/amx.py` explicitly guards BF16 on AMX/AVX2 support. CLI BF16
existence alone does not establish Kunpeng support. LLAMAFILE F32 remains the
bounded fallback requiring actual512/C64 local operator execution.
