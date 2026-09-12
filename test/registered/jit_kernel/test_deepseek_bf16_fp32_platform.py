from unittest.mock import patch

import torch

from sglang.jit_kernel.deepseek_v4 import _dispatch_bf16_fp32_backend


def test_non_cuda_bf16_fp32_linear_skips_cublas_jit():
    x = torch.randn(2, 8, dtype=torch.bfloat16)
    weight = torch.randn(4, 8, dtype=torch.bfloat16)

    with patch(
        "sglang.jit_kernel.deepseek_v4._jit_torch_cublas_bf16_fp32",
        side_effect=AssertionError("CUDA JIT must not run for a non-CUDA tensor"),
    ):
        actual = _dispatch_bf16_fp32_backend(x, weight, algo="cublas")

    expected = torch.nn.functional.linear(x.float(), weight.float())
    torch.testing.assert_close(actual, expected)
    assert actual.dtype == torch.float32
