import json
import tempfile
import unittest
from pathlib import Path

from round18.nz29_slot_writer_audit import audit


class Nz29SlotWriterAuditTests(unittest.TestCase):
    def test_distinguishes_parent_path_from_tensor_list_seam(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sources = {
                "vllm_ascend/ops/fused_moe/fused_moe.py": "\n".join(
                    (
                        "layer.w13_weight.data = maybe_trans_nz(layer.w13_weight.data)",
                        "layer.w13_weight_list = [weight.clone() for weight in layer.w13_weight.data.unbind(dim=0)]",
                        "w1 = w13_weight_list if isinstance(w13_weight_list, list) else layer.w13_weight",
                    )
                ),
                "vllm_ascend/eplb/adaptor/vllm_adaptor.py": "\n".join(
                    (
                        "buffer_tensor = torch.empty_like(expert_tensor)",
                        "expert_tensor.copy_(buffer_tensor)",
                        "self.log2phy_map_per_layer[layer_id].copy_(updated_log2phy_map)",
                    )
                ),
                "vllm_ascend/eplb/core/eplb_device_transfer_loader.py": "\n".join(
                    (
                        "local_expert_to_replace, buffer_tensor_id = recv_expert_info",
                        "self.eplb_adaptor.do_update_expert_weight",
                    )
                ),
                "vllm_ascend/eplb/core/eplb_utils.py": (
                    'assert not eplb_enable, "EPLB must used in expert parallelism."'
                ),
                "vllm_ascend/model_executor/offloader/prefetch.py": "\n".join(
                    (
                        "torch_npu.npu_format_cast(buffer, ACL_FORMAT_FRACTAL_NZ)",
                        "buffer_pool._buffers[key]",
                    )
                ),
            }
            for relative, text in sources.items():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8")
            index_copy = root / "IndexCopyKernelNpu.cpp"
            index_copy.write_text(
                "contiguous_self = npu_utils::format_contiguous(self);\n"
                "npu_utils::format_fresh_view(self, result);\n",
                encoding="utf-8",
            )
            npu_utils = root / "NpuUtils.cpp"
            npu_utils.write_text("x.copy_(y);\n", encoding="utf-8")
            round9 = root / "findings.md"
            round9.write_text(
                "shape 是完整 32-slot parent\n完整 parent→parent\n", encoding="utf-8"
            )
            receipt = root / "receipt.json"
            receipt.write_text(
                json.dumps(
                    {
                        "status": "FAIL_FIXED_SLOT_POSTLOAD_FORMAT_INELIGIBLE",
                        "mutation_started": False,
                        "postload_backing": {
                            "w13": {"npu_format": 29},
                            "w2": {"npu_format": 29},
                        },
                    }
                ),
                encoding="utf-8",
            )
            result = audit(
                source_root=root,
                index_copy_source=index_copy,
                npu_utils_source=npu_utils,
                round9_findings=round9,
                v018_receipt=receipt,
            )
            self.assertEqual(
                result["current_q32_parent"]["disposition"],
                "BLOCK_CURRENT_PARENT_FIXED_SLOT_PATH",
            )
            self.assertTrue(result["alternate_tensor_list_seam"]["exists"])
            self.assertFalse(result["alternate_tensor_list_seam"]["drop_in_for_current_s0"])


if __name__ == "__main__":
    unittest.main()
