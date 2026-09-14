# SGLang KT Round 3 Commits

Repository: `https://github.com/kvcache-ai/sglang.git`

Base: `0f36b26d`

Final: `06b319dc3a62b77c880e36b042d273bfc3957d12`

| Commit | Purpose |
|---|---|
| `a3ef6a76c` | Guard CUDA capability probe during NPU import |
| `cb862d608` | Avoid CUDA JIT for router linear on NPU |
| `8826a5cee` | Enable KTEP accelerator stream path on Ascend |
| `307acda4f` | Preserve KTEP dual-stream overlap and ordering |
| `691ba5a6f` | Skip KTEP wrapping for all-accelerator layers |
| `4002e16ff` | Normalize CPU expert mapping to contiguous int32 |
| `dd495ea4b` | Derive Ascend grouped-MoE expert count from physical weights |
| `06b319dc3` | Zero and sanitize CPU-owned `-1` routes before Ascend routing; add regression |

Patch preservation: `migration_bridge/round3/sglang_patches/` contains a numbered
`git format-patch` series from the frozen child base to the final child commit.

