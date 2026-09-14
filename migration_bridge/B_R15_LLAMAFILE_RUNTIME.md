# B Round 1.5 — LLAMAFILE ARM Runtime

## Result

**PASS — single-layer routed expert core runtime**

The current KML-off extension performed 1,000 deterministic single-token routed-expert forwards on Kunpeng 920 and matched a Torch CPU BF16-output reference exactly for the tested fixture.

## Test design

The test intentionally avoids a model, accelerator, SGLang, and KML. It directly instantiates the production C++ binding used underneath `LlamafileMoEWrapper`:

```python
config = kt_kernel_ext.moe.MOEConfig(...)
config.pool = cpuinfer.backend_
config.gate_proj = gate.data_ptr()
config.up_proj = up.data_ptr()
config.down_proj = down.data_ptr()
moe = kt_kernel_ext.moe.MOE(config)
```

It then loads weights through a CPUInfer task and executes `forward_task` with one selected expert and a routing weight of `0.75`.

Reference equation:

```text
y = routing_weight * down(silu(gate(x)) * up(x))
```

The comparison rounds the Torch reference to BF16 before converting it back to F32 for error measurement, matching the extension's output contract.

## Fixture

```text
seed=20260827
batch=1
hidden_size=32
intermediate_size=256
expert_num=2
top_k=1
weight_type=F32
hidden/output_type=BF16
worker subpools=1
worker threads=4
iterations=1000
```

## Output

| Metric | Value |
|---|---:|
| status | PASS |
| iterations | 1000 |
| all finite | true |
| max absolute error | 0.0 |
| mean absolute error | 0.0 |
| relative L2 error | 0.0 |
| elapsed | 0.024799656 s |
| smoke latency | 24.7997 µs/iteration |

The latency value is diagnostic only. It excludes wrapper/model overhead and uses a tiny tensor; it is not a KTransformers performance result.

## Evidence

- Test source: `migration_bridge/experiments/r15_cpu_llamafile_smoke.py`.
- A3 output: `/home/admin/kt_ascend_round1_c40d37c/logs/r15/cpuinfer-llamafile-smoke.log`.
- Build proof: the same log directory's `compile_commands.json` and `build.log`.

## What this proves

- the current generic MoE pybind class is usable with KML disabled;
- CPUInfer load/forward tasks and synchronization work on A3;
- the selected aarch64 LLAMAFILE calculation path produces correct results for the fixture;
- repeated execution does not crash during 1,000 synchronous iterations.

## What this does not prove

- The fixture uses in-memory F32 weights, not a minimal on-disk GGUF file. `GGUFLoader` ingestion is not dynamically tested.
- Quantized expert types, prefill batches, multiple routed experts, multiple worker subpools, and multi-NUMA memory placement are not tested.
- The memory-policy calls were denied by the constrained container, so NUMA binding is not verified.
- No full model, NPU, SGLang, tensor parallelism, graph, deferred expert, or dynamic placement path was executed.

These boundaries are why the result authorizes a LLAMAFILE-first MVP, not a production-performance claim.
