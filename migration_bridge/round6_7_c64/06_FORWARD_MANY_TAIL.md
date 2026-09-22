# Tail-aware forward_many

F32 intermediate work uses ceiling task count and the actual last-block length:
`704 = 256 + 256 + 192`. Gate GEMM, up GEMM, activation and down-input conversion
all use that last-block length. Down GEMM still reduces the complete K=704.
Unsupported F32 SGEMM returns fail explicitly. Block-quantized unaligned execution
is not enabled.

The Layer17 operator smoke passed qlen 2/8/9/10/11/16/32/64 and three real decode
operands for experts 1 and 63, each with 10 identical repeated outputs.

The diagnostic prefill matrix stacks genuine captured decode rows to exercise kernel
shapes. It is **not** labeled as a captured contiguous prefill trajectory. Actual model
prefill is a separate canary gate. Expanded early/middle/late layer coverage is running.
