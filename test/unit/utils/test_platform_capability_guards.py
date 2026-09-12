from unittest.mock import patch

import torch

from sglang.srt.utils.common import mxfp8_block_convert_required


def test_mxfp8_probe_does_not_query_cuda_when_unavailable():
    mxfp8_block_convert_required.cache_clear()
    try:
        with (
            patch.object(torch.version, "hip", None),
            patch.object(torch.cuda, "is_available", return_value=False),
            patch.object(
                torch.cuda,
                "get_device_capability",
                side_effect=AssertionError("CUDA probe must not run"),
            ),
        ):
            assert mxfp8_block_convert_required() is False
    finally:
        mxfp8_block_convert_required.cache_clear()
