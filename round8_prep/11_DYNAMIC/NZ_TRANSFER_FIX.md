# Ascend Dynamic transposed-NZ transfer fix

## Root-cause boundary

The actual 40-layer model trace brackets the warning inside
`copy_experts_weights_bf16()`, between COPY_BEGIN and COPY_END, before the
post-update weight auditor. The implicated source and destination are
FRACTAL_NZ, transposed layouts: w13 [256/32,2048,1024] stride
[2097152,1,2048], w2 [256/32,512,2048] stride [1048576,1,512].
Expert slicing introduces a nonzero storage offset; the first observed offset
is 12582912. The original contiguous-layout prototype did not reproduce this.

Ascend's [OpCommand implementation](https://github.com/Ascend/pytorch/blob/master/torch_npu/csrc/framework/OpCommand.cpp)
emits this warning from InputWithoutContiguous when an input's storage offset
is nonzero. That source explains the condition, but does not prove a specific
model output was corrupted. No corruption claim is made here.

## Reproduction and minimal change

`tools/test_dynamic_nz_transfer.py` extracts the actual deployed BF16 copy
function, constructs transposed NZ BF16 operands, and changes resident identity
three times. Independent CPU input tensors define the expected destination.
The old code passes the value oracle but fails the no-offset-warning assertion.
The patched code passes both with identical BF16 values.

Only the NPU branch of BF16 expert migration changes:

1. Convert the entire source tensor (offset zero) to ND.
2. Select resident experts in ND using the selected logical IDs.
3. Copy the complete selected tensor into the existing destination tensor.

No raw pointer transfer, weight arithmetic, mask policy, stream ownership,
resident budget, or C64 change. Destination tensor identity and NZ format are
retained. Non-NPU behavior is unchanged. Temporary source-ND/selected buffers
must remain included in the final HBM audit.

Original deployed file SHA256:
`3e5d6a38c375ed6e8d538cd4e3eea3b8dad2d725313f8554fe5a1b3e2ff0fbb7`.
The fixed complete file is retained in `round8_prep/runtime/kt_ep_wrapper.py`.
Deployment is a separate `sglang-python-dynamic-v2` copy; the original Static
runtime and historical results are not rewritten.

## Verification state

- A3/.22 operator regression: old warning reproduced; fixed 3-update CPU
  identity oracle and no-warning assertion PASS.
- Actual model old-path attribution: warning bracketed within BF16 copy.
- Fixed full-model DEV: 120 updates (40 layers x3), generation and weight
  assertions PASS; zero offset warnings versus one in the unmodified actual
  path. All five pre-fix/post-fix DEV token arrays are exact.
- Clean restart and observer parity: independent hook-free process, five of
  five DEV token arrays exact, no offset warning. Raw-response audit PASS.
- DEV1024 is skipped by the user's explicit instruction, preserving v1 failure.
- No held-out or CMMLU fitting permitted. Candidate freeze precedes either.
