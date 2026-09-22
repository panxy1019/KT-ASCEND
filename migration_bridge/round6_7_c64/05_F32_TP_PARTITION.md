# Generic F32 TP partition

Implementation candidate: `97d8ad841` (not yet model-qualified at this checkpoint).

F32 multi-subpool configurations use equal 32-aligned partitions. For DSV2-Lite,
1408 becomes offsets `[0,704]`, lengths `[704,704]`. The native loader slices gate/up
rows and down columns, retaining the original full-matrix source stride.

This is not a hardcoded 704 exception. Invalid positive dimensions, unequal splits,
and non-32-aligned F32 slices fail. Quantized paths retain the existing 256-block guard.
Single-TP C32 retains the deployed 1408 behavior.

Local C++ unit tests cover 1408 with TP1/2/4, exact contiguous coverage, malformed
shapes and quantized block rejection. Result: **LOCAL_VERIFIED PASS**.

Native TP2 load capture on Layer17 reconstructs the actual gate/up/down weight slices
exactly against frozen BF16 source values expanded to F32. The expanded matrix selects
a reproducible expert (1 or 63) on each of 26 layers; final coverage remains pending.
