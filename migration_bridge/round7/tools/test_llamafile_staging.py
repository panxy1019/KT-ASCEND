import unittest
from types import SimpleNamespace
from bf16_context_compat import port_source
from llamafile_bf16_staging import LlamafileBF16Stager, validate_layout
from runtime_selection import select_sglang_runtime
from test_bf16_context_compat import SOURCE


PIPELINE=SOURCE+'''        tp_world_size = 1
        do_write = tp_rank == 0 and wrapper is not None
        if do_write:
            def submit_write_expert(expert_id, slot):
                wrapper.submit_write_weight_scale_to_buffer(expert_id, slot)
            submit_write_expert(1, 0)
        for idx in range(3):
            if do_write:
                wrapper.sync_write_weight_scale_to_buffer()
                if idx > 0:
                    event.synchronize()
                submit_write_expert(1, idx % 2)
'''


class StagingTests(unittest.TestCase):
    def test_tp1_layout_only(self):
        shapes={'w13_weight':(2,3,8),'w2_weight':(2,4,3)}
        validate_layout(64,3,4,shapes,1)
        with self.assertRaisesRegex(ValueError,'TP1_ONLY'):
            validate_layout(64,3,4,shapes,2)
        with self.assertRaisesRegex(ValueError,'ASCEND_LAYOUT'):
            validate_layout(64,3,4,{'w13_weight':(2,8,3),'w2_weight':(2,3,4)},1)
    def test_index_rejected_before_any_write(self):
        instance=LlamafileBF16Stager.__new__(LlamafileBF16Stager)
        instance.num_experts=64
        for expert,slot in ((-1,0),(64,0),(0,2)):
            with self.assertRaisesRegex(ValueError,'STAGING_INDEX'):
                instance.submit(expert,slot)
    def test_port_keeps_native_fallback_and_event_fences(self):
        updated=port_source(PIPELINE,llamafile_staging=True)
        self.assertIn('wrapper.submit_write_weight_scale_to_buffer(expert_id, slot)',updated)
        self.assertIn('llamafile_writer.submit(expert_id, slot)',updated)
        self.assertIn('llamafile_writer.sync()',updated)
        self.assertIn('wrapper.sync_write_weight_scale_to_buffer()',updated)
        self.assertEqual(updated.count('event.synchronize()'),PIPELINE.count('event.synchronize()'))
        self.assertEqual(updated.count('return torch.cuda.current_stream()'),1)
    def test_optional_seam_rejects_drift(self):
        with self.assertRaisesRegex(AssertionError,'BF16_WRITER_SOURCE_DRIFT'):
            port_source(SOURCE,llamafile_staging=True)
        self.assertNotIn('llamafile_writer',port_source(PIPELINE))
    def test_helper_hash_binding(self):
        root='/workspace/c64-dev/round7/new/sglang-python'
        wrapper=root+'/sglang/srt/layers/moe/kt_ep_wrapper.py'
        helper=root+'/sglang/srt/layers/moe/llamafile_bf16_staging.py'
        provenance=dict(sglang_runtime_root=root,files={wrapper:{'sha256':'wrapper'},helper:{'sha256':'helper'}},
            bf16_context_port={'ported_sha256':'wrapper','staging_helper_sha256':'helper'})
        self.assertEqual(select_sglang_runtime({'sglang_runtime_root':root},provenance),root)
        provenance['files'][helper]['sha256']='drift'
        with self.assertRaisesRegex(ValueError,'PRIVATE_STAGING_HELPER_HASH_BINDING'):
            select_sglang_runtime({'sglang_runtime_root':root},provenance)
