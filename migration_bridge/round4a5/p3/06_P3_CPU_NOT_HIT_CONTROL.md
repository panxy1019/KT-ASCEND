# P3 CPU-Not-Hit Exact Control

Evidence state: `A3_VERIFIED_READY`

Real production captures at Layer 17 yielded 1,150 unique positions whose
top-k routes contained none of CPU-owned experts `{6,8,25,36}`. The frozen
minimum sample of 32 positions was checked.

| Invariant | Result |
|---|---|
| CPU partial is exact zero | 32/32 |
| Hybrid routed merge equals NPU partial exactly | 32/32 |
| routing inputs and weights finite | PASS |

This proves that merely enabling the eight-layer Hybrid infrastructure does
not contaminate a position where the CPU-owned route is absent. Artifact
`evidence/g5-cpu-not-hit-l17.json`, file SHA
`8c675f8b4a811759f10eb1ddcbdaa4d0f7325f2958ab1e30572ffbcce441a4db`.

`P3_CPU_NOT_HIT_CONTROL = A3_VERIFIED_READY`.
