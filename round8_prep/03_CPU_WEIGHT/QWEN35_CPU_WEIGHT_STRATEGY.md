# CPU weight strategy — two bounded candidates

Primary model BF16, no FP8 substitution. CPU BF16 native path is preferred only if Kunpeng probe proves supported arithmetic and performance. Official x86 BF16 evidence is not transferable.

Fallback: LLAMAFILE F32 GGUF with values exact-expanded from frozen BF16. Full routed pool is **120GiB payload** (40×256×3×2048×512×4); no claim this fits total process RSS. BF16 pool payload60GiB. Original checkpoint totals71,903,878,016bytes; checkpoint+F32 pool alone ~186.97GiB disk before conversion scratch, metadata and free-space reserve.

Index inspection finds 80 packed routed tensor entries across40 text layers (`gate_up_proj`, `down_proj`) and768 separate MTP expert entries, which must be excluded with MTP OFF. The index has no tensor shapes. Never assume packed gate/up order or transpose axes merely from names.

Before full conversion: read safetensors headers, freeze shapes/dtypes/offsets, extract one early/middle/late expert, prove gate/up split and down orientation against source, exact BF16→F32 identity, route weighting once. Then same-operand NPU / production CPU / canonical reference, decode+prefill, relL2≤0.01, repeat determinism. New dimensions512 and256-expert layout need tests; DSV2 intermediate1408 split704+704 is NOT reusable unchanged.

Streaming conversion must keep peak memory bounded to one tensor/expert block, exclude MTP and vision from CPU expert artifacts, preserve source index mapping and write new manifest/hash. Full conversion waits resource admission and sample identity PASS. No actual expert tensors converted in prep.
