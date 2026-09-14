# B Round 2A Summary

## Repository

- base commit：`c40d37c63cb9b7d041c8489167b3b822961d12c5`
- final commit：documentation commit 后以当前 `git rev-parse HEAD` 为准
- branch：`feature/kt-arm-llamafile-cpu-plane`

## ARM Variant

- status：PASS
- changes：aarch64 build-time vendor、runtime detection、extension metadata 统一为 `arm`；ARM 无 x86 fallback；架构 mismatch fail-fast
- A3 actual：`arch=aarch64`, `variant=arm`
- regression：mock x86 AVX2 和 unknown fallback 均通过

## GGUF

- fixture：local-only deterministic F32, seed 20260827, 2 layers, default 8 experts/top2, hidden/intermediate 256
- reproducibility：two independent GGUF files byte-identical, SHA-256 `e2a275952dd738223b45d05c220626b6103cffbfe732dc08537ccfa18b736247`
- loader：production keys `blk.N.ffn_{gate,up,down}_exps.weight`
- wrapper：`KTMoEWrapper -> LlamafileMoEWrapper -> GGUFLoader -> CPUInfer -> MOE`
- status：PASS

## Routed MoE

- num_experts：4 and 8
- top_k：2
- mapping：identity and permutations including `[2,0,3,1]`
- decode：qlen=1 PASS
- prefill：qlen=2/8/32/64 PASS
- edges：weights 1/0、same/different experts across tokens PASS
- batch：backend uses flattened-token semantics; qlen=2 covers two flattened tokens

## CPUInfer

- 1000 loop：PASS, deterministic, RSS `329977856 -> 329977856`, delta 0
- create/destroy：PASS, at least 20 alternating same/different paths
- multi-layer：layer 0 and layer 1 isolated PASS
- threadpool：count 1 and 2 correctness PASS
- lifetime：weights remain valid after load sync and Python temporary release

## NUMA

- nodes visible：`[0,1,2,3,4,5,6,7]`
- mapping：1 pool `[0]/[4 threads]`; 2 pools `[0,1]/[2,2 threads]`
- observability：PASS through actual backend config + process affinity + sysfs nodes
- boundary：unprivileged container memory-bind enforcement not verified; no placement/performance claim

## Numerical

- dtype：BF16 I/O, F32 fixture weights
- max_abs_error：`6.103515625e-05` (limit `1e-3`)
- mean_abs_error：`1.4901161193847656e-08` (limit `1e-4`)
- relative_l2_error：`3.9346272514744805e-05` (limit `1e-2`)

## ARM ISA

- current required baseline：`armv8.2-a+fp16+dotprod+sve+bf16` because CMake applies it globally
- A3 features：compatible; build/import/all correctness tests PASS
- remaining portability risks：`P1_BUILD_PORTABILITY_RISK`; do not claim compatibility with every Kunpeng 920/generic aarch64 CPU
- i8mm：host supports it, active build flag does not require it

## A3

- CPU-only build：PASS
- import：PASS
- test matrix：`21 passed in 4.82s`
- no KML：confirmed OFF
- no NPU：no devices, backend autoload disabled, `torch=2.9.0+cpu`
- host untouched：dependencies installed only in disposable container
- evidence retained：`/home/admin/kt_round2a_c40d37c/logs/round2a/`

## Exit Gate

```text
CPU_EXPERT_PLANE = A3_VERIFIED_READY
```

所有 Round 2A mandatory correctness、lifecycle、observability 和 isolation gates 已通过。

## Remaining blockers

Round 2A / A3 CPU Expert Plane 无 blocker。进入通用 ARM 发布前仍有 ISA portability P1；进入 Round 2B 后才允许处理 Ascend runtime plane。NUMA membind 和性能没有在本轮验证，也不应被当作已完成。

## Commits

1. `734da50` fix(cpu): report native ARM variant on aarch64
2. `aba93a3` test(llamafile): add deterministic GGUF MoE fixture
3. `4f9a5cc` fix(llamafile): support CPU wrapper execution and expert mapping
4. `ea61b53` test(moe): cover ARM routed expert and prefill correctness
5. `7377568` test(cpuinfer): harden LLAMAFILE lifecycle and NUMA coverage
6. `e2af3ee` fix(cpu): allow wrappers without a pinned allocator
7. `0cfb771` fix(llamafile): enforce the prefill hidden-size contract
8. `4fe8668` test(moe): apply BF16 numerical acceptance thresholds
9. documentation commit containing `migration_bridge/round2a/`
