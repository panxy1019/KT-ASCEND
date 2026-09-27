import importlib, os, sys, traceback
snapshot='/snapshot'; sys.path.insert(0,snapshot+'/overlay-python')
print('STEP import torch',flush=True); import torch
print('STEP import torch_npu',flush=True); import torch_npu
print('DEVICE_ENV',repr(os.environ.get('ASCEND_RT_VISIBLE_DEVICES')),flush=True)
original=torch.npu.is_available
print('ORIGINAL_AVAILABLE',original(),flush=True)
torch.npu.is_available=lambda: True
modules=[
 'sglang.srt.server_args',
 'sglang.srt.layers.round9.ple_binding',
 'sglang.srt.layers.round9.qsa',
 'sglang.srt.mem_cache.round9_qsa_pool',
 'sglang.srt.models.qwen4_exp',
]
for name in modules:
 print('BEFORE',name,flush=True)
 importlib.import_module(name)
 print('AFTER',name,flush=True)
print('BEFORE ServerArgs construction',flush=True)
import json
ServerArgs=importlib.import_module('sglang.srt.server_args').ServerArgs
profile=json.load(open(snapshot+'/native-tp8-server-profile.json'))
args=ServerArgs(**profile)
print('AFTER ServerArgs construction',args.tp_size,args.device,flush=True)
torch.npu.is_available=original
print('RESTORED',original(),flush=True)
