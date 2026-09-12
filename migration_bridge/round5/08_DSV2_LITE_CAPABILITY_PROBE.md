# Round 5 — Raw Upstream DeepSeek-V2-Lite Capability Probe

Evidence state: `A3_VERIFIED` for the failures below.

## Frozen profile

```text
model: deepseek-ai/DeepSeek-V2-Lite
revision: 604d5664dddd88a0433dbae533b7fe9472482de0
TP: 1
NPU: NPU0 only
CPU backend: LLAMAFILE
CPU weights: frozen F32 GGUF
NPU weights: BF16 / unquantized
hidden size: 2048
MoE intermediate size: 1408
Graph/Deferred/Dynamic/MTP/Speculative: OFF
```

The frozen GGUF was read from the immutable Round 4A artifacts. Its file hash
remains `992d32e0de59e135464f77fc4869e810f3d71b3ec01f6ed0996cbeaa58e49df4`.

## Probe matrix

| Capability | Raw result | First evidence |
|---|---|---|
| BUILD | PASS | raw wheel |
| NPU tensor runtime | PASS | one visible NPU and exact smoke |
| KTEP import | FAIL | unguarded CUDA capability query |
| CPU expert GGUF metadata | PASS | 24 tensors loaded |
| CPU expert construction | FAIL | 1408 not divisible by 256 |
| NPU expert load | NOT RUN | prior import/CPU hard gates |
| static placement | NOT RUN | prior hard gates |
| prefill | NOT RUN | prior hard gates |
| decode | NOT RUN | prior hard gates |
| all-NPU serving | NOT RUN | SGLang import stack blocked first |

## Blocker 1 — SGLang CUDA capability guard

Importing the pinned upstream `KTEPWrapperMethod` on torch_npu traversed the
quantization registry and called:

```text
torch.cuda.get_device_capability()
```

from `mxfp8_block_convert_required()` even though CUDA is unavailable. It
failed with:

```text
AssertionError: Torch not compiled with CUDA enabled
```

The immutable local child contains a narrow guard that returns false when
`torch.cuda.is_available()` is false. This is a known local semantic fix, not a
new acceptance relaxation.

## Blocker 2 — DeepSeek-V2-Lite intermediate shape

The raw `LlamafileMoEWrapper` successfully opened the frozen P3 GGUF and read
its metadata, then rejected construction:

```text
ValueError: intermediate_size (1408) must be divisible by QK_K (256)
```

DeepSeek-V2-Lite's 1408 dimension is immutable. Round 4A already contains and
qualifies a remainder-safe one-threadpool implementation for this shape.

## Numerical blocker from static audit

Even after the two startup blockers, raw upstream lacks the Round4A.1
BF16-visible expert boundaries and the enabled unquantized FP32 Hybrid merge.
They must be ported and revalidated rather than discovered only at P1 token
qualification.

## Raw conclusion

```text
R5_G3_RAW_DSV2_LITE_CAPABILITY = FAIL
FAILURE_CLASS_1 = KTEP_IMPORT_CUDA_GUARD
FAILURE_CLASS_2 = LLAMAFILE_INTERMEDIATE_1408
RAW_PREFILL = NOT_RUN
RAW_DECODE = NOT_RUN
```

This result is not rewritten after alignment patches.

## Evidence

- `evidence/a3_raw_ktep_import.log`
- `evidence/a3_raw_dsv2_llamafile_capability.json`

