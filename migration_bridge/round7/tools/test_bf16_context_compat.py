import contextlib
from types import SimpleNamespace
import unittest
from bf16_context_compat import port_source, port_cpu_buffers
from runtime_selection import select_sglang_runtime


SOURCE='''class Context:
    def other(self):
        return torch.cuda.current_stream()
    def _prepare_weight_bf16(self, wrapper, original_layer=None, gpu_experts_mask=None,
                             logical_to_gpu_index=None):
        # Bind Python thread to specific CPU core (last cores for each rank)
        tp_rank = get_tensor_model_parallel_rank()
        num_cpus = os.cpu_count()
        target_cpu = num_cpus - 1 - tp_rank
        os.sched_setaffinity(0, {target_cpu})
        layer = self.gpu_layer
        device = layer.w13_weight.device
        copy_stream = torch.cuda.Stream(device=device)
        post_stream = torch.cuda.Stream(device=device)
        event = torch.cuda.Event()
        with torch.cuda.stream(copy_stream):
            event.record(copy_stream)
        with torch.cuda.stream(post_stream):
            post_stream.wait_event(event)
        with torch.cuda.stream(post_stream):
            event.synchronize()
        torch.cuda.current_stream(device).wait_stream(post_stream)
'''


class ContextCompatTests(unittest.TestCase):
    def test_transport_selects_actual_device_without_rebinding(self):
        calls=[]
        class Stream:
            def __init__(self,device=None):calls.append(('stream',device))
            def wait_event(self,event):calls.append('wait_event')
            def wait_stream(self,stream):calls.append('wait_stream')
        class Event:
            def record(self,stream):calls.append('record')
            def synchronize(self):calls.append('event_sync')
        api=SimpleNamespace(Stream=Stream,Event=Event,stream=lambda stream:contextlib.nullcontext(),
            current_stream=lambda device:Stream(device))
        def select(kind):calls.append(('selected_device',kind));return api
        def forbidden(*args):raise AssertionError('CPUSET_MUTATED')
        namespace=dict(torch=SimpleNamespace(get_device_module=select),
            os=SimpleNamespace(sched_setaffinity=forbidden),get_tensor_model_parallel_rank=lambda:0)
        exec(port_source(SOURCE),namespace)
        instance=namespace['Context']()
        instance.gpu_layer=SimpleNamespace(w13_weight=SimpleNamespace(device=SimpleNamespace(type='npu')))
        instance._prepare_weight_bf16(None)
        self.assertIn(('selected_device','npu'),calls)
        self.assertIn('wait_stream',calls)
    def test_other_cuda_method_unchanged(self):
        self.assertIn('return torch.cuda.current_stream()',port_source(SOURCE))
    def test_resident_phase_port_is_scoped_and_guarded(self):
        phase='        # --- Phase 1: Copy GPU experts directly (fast GPU-to-GPU) ---\n'
        copy='                    dst[e].copy_(src[gpu_idx], non_blocking=True)\n'
        source=SOURCE.replace('        copy_stream = torch.cuda.Stream(device=device)\n',
            '        gpu_expert_ids = []\n        weight_infos = []\n'+phase+
            '        for e in gpu_expert_ids:\n            for name, _, dst in weight_infos:\n'
            '                    src = getattr(original_layer, name)\n'+copy+
            '        copy_stream = torch.cuda.Stream(device=device)\n')
        output=port_source(source,resident_layout=True)
        self.assertIn('resident_to_raw(getattr(original_layer, name), dst)',output)
        self.assertIn('if device.type == "npu":\n                        src = resident_sources[name]',output)
        self.assertIn('return torch.cuda.current_stream()',output)
        with self.assertRaisesRegex(AssertionError,'RESIDENT_PHASE_SOURCE_DRIFT'):
            port_source(SOURCE,resident_layout=True)
    def test_npu_bf16_allocator_owned_pinned_and_tp1(self):
        source='''class Context:
    def _create_cpu_buffers(self):
        for name in self.weight_names:
                gpu_tensor = getattr(self.gpu_layer, name)
                expert_shape = gpu_tensor.shape[1:]
                buf_dtype = gpu_tensor.dtype
                element_size = torch.empty((), dtype=buf_dtype).element_size()
                self.cpu_buffers[name] = 'original_shared_memory_path'
'''
        calls=[]
        tensor=SimpleNamespace(shape=(64,12,4),dtype='bf16',device=SimpleNamespace(type='npu'))
        def empty(shape, **kwargs):
            calls.append((shape,kwargs))
            return SimpleNamespace(element_size=lambda:2)
        namespace=dict(torch=SimpleNamespace(empty=empty), get_tensor_model_parallel_world_size=lambda:1)
        exec(port_cpu_buffers(source), namespace)
        context=namespace['Context']()
        context.weight_names=['w13_weight'];context.gpu_layer=SimpleNamespace(w13_weight=tensor)
        context._is_bf16_quant=True;context.cpu_buffers={}
        context._create_cpu_buffers()
        self.assertEqual(calls,[((2,12,4),dict(dtype='bf16',device='cpu',pin_memory=True))])
        namespace['get_tensor_model_parallel_world_size']=lambda:2
        with self.assertRaisesRegex(RuntimeError,'TP1_ONLY'):context._create_cpu_buffers()
        namespace['get_tensor_model_parallel_world_size']=lambda:1
        for bf16,device in ((False,'npu'),(True,'cuda')):
            context._is_bf16_quant=bf16;tensor.device.type=device
            context._create_cpu_buffers()
            self.assertEqual(context.cpu_buffers['w13_weight'],'original_shared_memory_path')
    def test_allocator_source_drift_rejected(self):
        with self.assertRaisesRegex(AssertionError,'METHOD_AMBIGUOUS'):port_cpu_buffers(SOURCE)
    def test_source_drift_rejected(self):
        with self.assertRaisesRegex(AssertionError,'BF16_AFFINITY_SOURCE_DRIFT'):
            port_source(SOURCE.replace('num_cpus = os.cpu_count()','num_cpus = 1'))
        with self.assertRaisesRegex(AssertionError,'BF16_STREAM_SOURCE_DRIFT'):
            port_source(SOURCE.replace('torch.cuda.Event()','different.Event()'))
    def test_runtime_path_hash_binding(self):
        root='/workspace/c64-dev/round7/new/sglang-python'
        wrapper=root+'/sglang/srt/layers/moe/kt_ep_wrapper.py'
        cfg=dict(sglang_runtime_root=root)
        provenance=dict(sglang_runtime_root=root,files={wrapper:{'sha256':'new'}},
            bf16_context_port={'ported_sha256':'new'})
        self.assertEqual(select_sglang_runtime(cfg,provenance),root)
        with self.assertRaisesRegex(ValueError,'PRIVATE_SGLANG_NOT_SELECTED'):
            select_sglang_runtime({},provenance)
        with self.assertRaisesRegex(ValueError,'PRIVATE_SGLANG_OUTSIDE_OWNED_SCOPE'):
            select_sglang_runtime({'sglang_runtime_root':'/tmp/runtime'},provenance)
        provenance['bf16_context_port']['ported_sha256']='drift'
        with self.assertRaisesRegex(ValueError,'PRIVATE_SGLANG_HASH_BINDING'):
            select_sglang_runtime(cfg,provenance)
