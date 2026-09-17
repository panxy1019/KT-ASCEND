"""Scoped source port for BF16 full-context transport; no model math edits."""
import ast


def port_cpu_buffers(source):
    """Keep CUDA/quantized shared-memory paths; BF16 NPU TP1 owns pinned tensors."""
    tree = ast.parse(source)
    methods = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
               and node.name == '_create_cpu_buffers']
    assert len(methods) == 1, 'BF16_CPU_BUFFER_METHOD_AMBIGUOUS'
    method = methods[0]
    lines = source.splitlines(keepends=True)
    original = ''.join(lines[method.lineno-1:method.end_lineno])
    anchor = '                element_size = torch.empty((), dtype=buf_dtype).element_size()\n'
    assert original.count(anchor) == 1, 'BF16_CPU_BUFFER_SOURCE_DRIFT'
    updated = original.replace(anchor,
        '                # Framework-owned pinned buffers participate in NPU copy lifetime.\n'
        '                if self._is_bf16_quant and gpu_tensor.device.type == "npu":\n'
        '                    if get_tensor_model_parallel_world_size() != 1:\n'
        '                        raise RuntimeError("BF16_NPU_PINNED_BUFFERS_TP1_ONLY")\n'
        '                    self.cpu_buffers[name] = torch.empty(\n'
        '                        (2,) + expert_shape, dtype=buf_dtype,\n'
        '                        device="cpu", pin_memory=True,\n'
        '                    )\n'
        '                    continue\n' + anchor)
    return ''.join(lines[:method.lineno-1])+updated+''.join(lines[method.end_lineno:])


def port_source(source, llamafile_staging=False):
    tree=ast.parse(source)
    matches=[node for node in ast.walk(tree) if isinstance(node,ast.FunctionDef)
             and node.name=='_prepare_weight_bf16']
    assert len(matches)==1,'BF16_METHOD_AMBIGUOUS'
    method=matches[0];lines=source.splitlines(keepends=True)
    original=''.join(lines[method.lineno-1:method.end_lineno])
    pin=('        # Bind Python thread to specific CPU core (last cores for each rank)\n'
         '        tp_rank = get_tensor_model_parallel_rank()\n'
         '        num_cpus = os.cpu_count()\n'
         '        target_cpu = num_cpus - 1 - tp_rank\n'
         '        os.sched_setaffinity(0, {target_cpu})\n')
    assert original.count(pin)==1,'BF16_AFFINITY_SOURCE_DRIFT'
    updated=original.replace(pin,
        '        # Preserve the caller/container cpuset; do not pin outside it.\n'
        '        tp_rank = get_tensor_model_parallel_rank()\n')
    device='        device = layer.w13_weight.device\n'
    assert updated.count(device)==1,'BF16_DEVICE_SOURCE_DRIFT'
    updated=updated.replace(device,device+'        device_api = torch.get_device_module(device.type)\n')
    assert updated.count('torch.cuda.')==7,'BF16_STREAM_SOURCE_DRIFT'
    updated=updated.replace('torch.cuda.','device_api.')
    if llamafile_staging:
        writer='        do_write = tp_rank == 0 and wrapper is not None\n'
        assert updated.count(writer)==1,'BF16_WRITER_SOURCE_DRIFT'
        updated=updated.replace(writer,writer+
            '        llamafile_writer = None\n'
            '        if do_write and not callable(getattr(wrapper, "submit_write_weight_scale_to_buffer", None)):\n'
            '            from .llamafile_bf16_staging import LlamafileBF16Stager\n'
            '            llamafile_writer = LlamafileBF16Stager(wrapper, self.cpu_buffers, tp_world_size)\n')
        submit='            def submit_write_expert(expert_id, slot):\n'
        assert updated.count(submit)==1,'BF16_SUBMIT_SOURCE_DRIFT'
        updated=updated.replace(submit,submit+
            '                if llamafile_writer is not None:\n'
            '                    llamafile_writer.submit(expert_id, slot)\n'
            '                    return\n')
        sync='                wrapper.sync_write_weight_scale_to_buffer()\n'
        assert updated.count(sync)==1,'BF16_SYNC_SOURCE_DRIFT'
        updated=updated.replace(sync,
            '                if llamafile_writer is not None:\n'
            '                    llamafile_writer.sync()\n'
            '                else:\n'+ '    '+sync)
    result=''.join(lines[:method.lineno-1])+updated+''.join(lines[method.end_lineno:])
    if llamafile_staging:
        result = port_cpu_buffers(result)
    compile(result,'kt_ep_wrapper.py','exec')
    return result
