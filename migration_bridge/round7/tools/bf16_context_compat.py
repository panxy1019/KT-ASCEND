"""Scoped source port for BF16 full-context transport; no model math edits."""
import ast


def port_source(source):
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
    result=''.join(lines[:method.lineno-1])+updated+''.join(lines[method.end_lineno:])
    compile(result,'kt_ep_wrapper.py','exec')
    return result
