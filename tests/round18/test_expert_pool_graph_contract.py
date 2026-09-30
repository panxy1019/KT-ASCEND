import tempfile
import unittest
from pathlib import Path

from round18.expert_pool_graph_contract import audit_frozen_graph_pool_contract


ACL_GRAPH = """
class ACLGraphWrapper:
    def __call__(self, *args, **kwargs):
        if entry.aclgraph is None:
            output = self.runnable(*args, **kwargs)
            return output
        torch.npu.current_stream().synchronize()
        entry.aclgraph.replay()
        return entry.output
"""

MODEL_RUNNER = """
if self.dynamic_eplb:
    self.eplb_updator.forward_before()
hidden_states = self._model_forward()
if self.dynamic_eplb:
    self.eplb_updator.forward_end(self.eplb_heat_collection_status)
"""

ADAPTOR = """
def do_update_expert_map(self, layer_id, updated_expert_map):
    self.expert_map_per_layer_cpu[layer_id].copy_(updated_expert_map)

def do_update_expert_weight(self, layer_id, local_expert_to_replace, buffer_tensor_id):
    expert_tensor.copy_(buffer_tensor)

def do_update_log2phy_map(self, layer_id, updated_log2phy_map):
    self.log2phy_map_per_layer[layer_id].copy_(updated_log2phy_map)
"""

LOADER = """
def update_expert_map_and_weight(self, reqs):
    for req in reqs:
        req.wait()
    self.eplb_adaptor.do_update_expert_map(self.layer_id, self.updated_expert_map)
    self.eplb_adaptor.do_update_log2phy_map(self.layer_id, self.updated_log2phy_map)
    self.eplb_adaptor.do_update_expert_weight(self.layer_id, 0, 0)
"""

EPLB_UTILS = """
def init_eplb_config(eplb_enable, ep_size):
    if ep_size == 1:
        assert not eplb_enable, "EPLB must used in expert parallelism."
"""

FUSED_MOE = """
fused_experts_results = self._quant_method.apply(
    expert_map=self._expert_map,
    log2phy=self.log2phy,
)
"""


class ExpertPoolGraphContractTests(unittest.TestCase):
    def _source_tree(self, *, loader=LOADER) -> Path:
        root = Path(self.temp_dir.name)
        files = {
            "vllm_ascend/compilation/acl_graph.py": ACL_GRAPH,
            "vllm_ascend/worker/model_runner_v1.py": MODEL_RUNNER,
            "vllm_ascend/eplb/adaptor/vllm_adaptor.py": ADAPTOR,
            "vllm_ascend/eplb/core/eplb_device_transfer_loader.py": loader,
            "vllm_ascend/eplb/core/eplb_utils.py": EPLB_UTILS,
            "vllm_ascend/ops/fused_moe/fused_moe.py": FUSED_MOE,
        }
        for relative, content in files.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        return root

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_complete_source_contract_allows_only_bounded_device_probe(self) -> None:
        result = audit_frozen_graph_pool_contract(self._source_tree())

        self.assertEqual(result["disposition"], "PROCEED_TO_BOUNDED_NPU_COMPONENT_PROBE")
        self.assertFalse(result["claims"]["npu_pass"])
        self.assertTrue(result["full_replay"]["replay_skips_python_runnable"])
        self.assertTrue(result["full_replay"]["current_stream_sync_before_replay"])
        self.assertTrue(result["publish_path"]["weight_copy_is_in_place"])
        self.assertTrue(result["publish_path"]["wait_precedes_publish"])
        self.assertTrue(result["publish_path"]["device_mapping_is_in_place"])
        self.assertTrue(result["native_consumer"]["reads_log2phy"])
        self.assertEqual(result["limits"]["current_token_route_miss_fill"], "MISSING")
        self.assertEqual(result["limits"]["ep1_existing_eplb_path"], "REJECTED_BY_FROZEN_SOURCE")

    def test_missing_wait_closes_gate_instead_of_inferring_safe_publish(self) -> None:
        result = audit_frozen_graph_pool_contract(
            self._source_tree(loader=LOADER.replace("    for req in reqs:\n        req.wait()\n", ""))
        )

        self.assertEqual(result["disposition"], "BLOCKED_MISSING_SOURCE_PRIMITIVE")
        self.assertFalse(result["publish_path"]["wait_precedes_publish"])
        self.assertIn("transfer_completion_before_publish", result["missing_primitives"])

    def test_audit_rejects_incomplete_frozen_source_identity(self) -> None:
        root = self._source_tree()
        (root / "vllm_ascend/eplb/core/eplb_utils.py").unlink()

        result = audit_frozen_graph_pool_contract(root)

        self.assertEqual(result["disposition"], "BLOCKED_INCOMPLETE_FROZEN_SOURCE")
        self.assertEqual(
            result["missing_source_files"],
            ["vllm_ascend/eplb/core/eplb_utils.py"],
        )


if __name__ == "__main__":
    unittest.main()
