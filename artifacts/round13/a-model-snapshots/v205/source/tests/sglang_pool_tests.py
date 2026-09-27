"""Real f02 pool lifecycle with bounded synthetic states, no model weights."""
import unittest
from unittest.mock import patch
from types import SimpleNamespace as NS
import torch
import torch_npu
torch.npu.set_device(0)
torch.set_num_threads(2)
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool


class PoolLifecycleTests(unittest.TestCase):
    def make_pool(self):
        params = NS(shape=NS(conv=[(8, 3)], temporal=(2, 4, 4)),
                    dtype=NS(conv=torch.bfloat16, temporal=torch.float32), layers=[0, 2])
        return HybridReqToTokenPool(
            size=2, mamba_size=2, mamba_spec_state_size=2, max_context_len=32,
            device="npu:0", enable_memory_saver=False, cache_params=params,
            enable_mamba_extra_buffer=False, short_conv_layer_ids=[1],
            short_conv_state_shape=(16, 9), ngram_context_len=2, eos_token_id=99)

    def test_alloc_free_reuse_resets_all_sibling_states(self):
        pool = self.make_pool()
        first = pool.mamba_pool.alloc(2)
        pool.short_conv_layer_cache(1)[first] = 7
        pool.set_ngram_context(first, torch.tensor([[1, 2], [3, 4]], device="npu:0"))
        pool.mamba2_layer_cache(0).temporal[first] = 11
        pool.mamba_pool.free(first)
        reused = pool.mamba_pool.alloc(2)
        self.assertEqual(torch.count_nonzero(pool.short_conv_layer_cache(1)[reused]).item(), 0)
        self.assertTrue(torch.all(pool.get_ngram_context(reused) == 99).item())
        self.assertEqual(torch.count_nonzero(pool.mamba2_layer_cache(0).temporal[reused]).item(), 0)

    def test_fork_and_clear_keep_siblings_with_owner(self):
        pool = self.make_pool()
        source = pool.mamba_pool.alloc(1)
        pool.short_conv_layer_cache(1)[source] = 4
        pool.set_ngram_context(source, torch.tensor([[5, 6]], device="npu:0"))
        target = pool.mamba_pool.fork_from(source)
        self.assertTrue(torch.equal(pool.get_ngram_context(source), pool.get_ngram_context(target)))
        self.assertTrue(torch.equal(pool.short_conv_layer_cache(1)[source], pool.short_conv_layer_cache(1)[target]))
        pool.clear()
        new_slots = pool.mamba_pool.alloc(2)
        self.assertTrue(torch.all(pool.get_ngram_context(new_slots) == 99).item())
        self.assertEqual(torch.count_nonzero(pool.short_conv_layer_cache(1)[new_slots]).item(), 0)

    def test_request_abort_free_and_slot_reuse(self):
        pool = self.make_pool()
        class Req:
            req_pool_idx = None
            mamba_pool_idx = None
            def is_dllm(self): return False
        req = Req()
        req.req_pool_idx = pool.alloc([req])[0]
        state = pool.get_mamba_indices(torch.tensor([req.req_pool_idx], device="npu:0"))
        pool.set_ngram_context(state, torch.tensor([[7, 8]], device="npu:0"))
        pool.free_mamba_cache(req)
        pool.free(req)
        new = Req()
        new.req_pool_idx = pool.alloc([new])[0]
        slot = pool.get_mamba_indices(torch.tensor([new.req_pool_idx], device="npu:0"))
        self.assertTrue(torch.all(pool.get_ngram_context(slot) == 99).item())


class QSAPoolTests(unittest.TestCase):
    def make_pool(self, page_size=4, enable_memory_saver=False):
        from sglang.srt.mem_cache.round9_qsa_pool import Round9QSAKVPool
        return Round9QSAKVPool(
            size=32, dtype=torch.bfloat16, page_size=page_size, head_num=2,
            head_dim=8, full_attention_layer_ids=[3, 7],
            enable_kvcache_transpose=False, device="npu:0", mamba_pool=None,
            enable_memory_saver=enable_memory_saver,
            request_capacity=2, index_head_dim=8, compress_ratio=4)

    def test_compressed_buffers_layer_and_request_isolation(self):
        pool = self.make_pool()
        loc = torch.tensor([4, 5], device="npu:0")
        values = torch.ones(2, 1, 8, device="npu:0", dtype=torch.bfloat16)
        pool.set_qsa_key_state_buffer(3, loc, values)
        pool.set_qsa_rope_position_buffer(loc, torch.tensor([0, 1], device="npu:0"))
        self.assertEqual(pool.get_qsa_key_state_buffer(3)[loc].sum().item(), 16)
        self.assertEqual(pool.get_qsa_key_state_buffer(7).sum().item(), 0)
        pool.reset_qsa_request(3, 1)
        self.assertEqual(pool.get_qsa_key_state_buffer(3)[loc].sum().item(), 0)
        self.assertTrue(torch.all(pool.get_qsa_rope_position_buffer(loc) == -1).item())
        group = torch.tensor([2], device="npu:0")
        pool.set_qsa_compressed_k_buffer(7, group, values[:1])
        self.assertEqual(pool.get_qsa_compressed_k_buffer(7)[group].sum().item(), 8)
        self.assertEqual(pool.get_qsa_compressed_k_buffer(3).sum().item(), 0)

    def test_requires_compression_aligned_pages(self):
        for page in (1, 2, 6):
            with self.subTest(page=page), self.assertRaises(ValueError):
                self.make_pool(page)
        with self.assertRaises(ValueError):
            self.make_pool(enable_memory_saver=True)

    def test_last_physical_token_and_compressed_group_are_addressable(self):
        pool = self.make_pool()
        locations = torch.tensor([4, 35], device="npu:0")
        keys = torch.arange(32, device="npu:0", dtype=torch.bfloat16).reshape(2, 2, 8)
        values = -keys
        pool.set_kv_buffer(NS(layer_id=3), locations, keys, values)
        self.assertTrue(torch.equal(pool.get_key_buffer(3).reshape(-1, 2, 8)[locations], keys))
        self.assertTrue(torch.equal(pool.get_value_buffer(3).reshape(-1, 2, 8)[locations], values))
        group = torch.tensor([8], device="npu:0")
        compressed = keys[:1, :1]
        pool.set_qsa_compressed_k_buffer(3, group, compressed)
        self.assertTrue(torch.equal(pool.get_qsa_compressed_k_buffer(3)[group], compressed))


class RunnerBudgetTests(unittest.TestCase):
    def make_runner(self):
        from sglang.srt.model_executor.model_runner_kv_cache_mixin import ModelRunnerKVCacheMixin
        runner = ModelRunnerKVCacheMixin()
        runner.mambaish_config = NS(model_type="qwen4_exp_text", indexer_head_dim=8,
            short_conv_state_shape=(16, 9), short_conv_layer_ids=[1],
            ngram_context_len=2, full_attention_layer_ids=[3, 7],
            mamba2_cache_params=NS(mamba_cache_per_req=1000))
        runner.kv_cache_dtype = torch.bfloat16
        runner.use_mla_backend = False
        runner.model_config = NS(hf_config=NS(architectures=["Qwen4ExpForConditionalGeneration"]), is_hybrid_swa=False,
            get_num_kv_heads=lambda tp: 2, head_dim=8, v_head_dim=8)
        runner.server_args = NS(max_mamba_cache_size=1, enable_dp_attention=False,
            disable_radix_cache=True, disable_cuda_graph=True, max_running_requests=1,
            disaggregation_mode="null", enable_memory_saver=False)
        runner.spec_algorithm = NS(is_none=lambda: True)
        runner.page_size = 4
        return runner

    def test_budget_includes_all_side_buffers_and_padding(self):
        runner = self.make_runner()
        with patch("sglang.srt.model_executor.model_runner_kv_cache_mixin.get_attention_tp_size", return_value=1):
            self.assertEqual(runner.get_cell_size_per_token(2), 136)
            remaining = runner.handle_max_mamba_cache(1.0)
        # Two Mamba slots, two PLE conv/history rows, QSA ring/positions,
        # one padded ordinary+compressed KV page; no silently free side buffers.
        expected = 2*1000 + 2*(16*9*2 + 2*8) + 4*(2*8*2 + 8) + 4*136
        self.assertEqual((1.0 - remaining) * 2**30, expected)

    def test_shared_profile_rejects_unbounded_request_count(self):
        runner = self.make_runner()
        runner.server_args.max_running_requests = 2
        with self.assertRaises(ValueError):
            runner.handle_max_mamba_cache(1.0)


if __name__ == "__main__":
    unittest.main()
