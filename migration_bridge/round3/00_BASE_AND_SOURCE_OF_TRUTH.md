# Round 3 Base and Source of Truth

## Frozen repositories

- KTransformers repository: `https://github.com/kvcache-ai/ktransformers.git`
- Round 2C frozen parent: `5b382086940f3c5f92bf52120fc1004e6d53026b`
- Round 3 branch: `feature/kt-deepseek-v2-lite-ascend-tp1`
- SGLang repository: `https://github.com/kvcache-ai/sglang.git`
- SGLang frozen base: `0f36b26d`
- SGLang Round 3 final: `06b319dc3a62b77c880e36b042d273bfc3957d12`
- SGLang branch: `feature/kt-ep-ascend-tp1`

The parent submodule pointer is authoritative. `migration_bridge/round3/sglang_patches/`
is a second, reviewable preservation mechanism and is not another source tree.

## Model source

- Model: DeepSeek-V2-Lite
- Exact Hugging Face snapshot revision: `604d5664`
- Full revision recorded by the A3 snapshot: `604d5664...`
- Model path in the disposable container: `/workspace/models/DeepSeek-V2-Lite-604d5664`
- `config.json` SHA256: `f346286b0f1c8b044252fd54cb4fa78b9fab6472a6e8bebb9edfe03d414ea03d`
- `tokenizer.json` SHA256: `41f3bf64213da8c012d8bd0871a58a1fdf70463e8f08f110ddbb1082f529f669`

The NPU weights and the exported CPU expert tensors came from this same snapshot.
No legacy Ascend operator implementation was copied into the result.

## Execution boundary

All builds, dependency installation, model execution, and tests ran in the A3
disposable container `kt-r3-dsv2lite`. The host OS and business containers were
not modified. The container was restricted to NPU 0, CPUs 0-15, and 128 GiB RAM.

