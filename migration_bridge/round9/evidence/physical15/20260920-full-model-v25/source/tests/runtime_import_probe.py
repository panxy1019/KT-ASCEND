"""Import/config gate only: no model construction, loading or inference."""
import argparse
import hashlib
import importlib
import json
import pathlib
import traceback
import faulthandler
faulthandler.enable()
faulthandler.dump_traceback_later(60,repeat=True)
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--model',type=pathlib.Path,required=True)
p.add_argument('--output',type=pathlib.Path,required=True)
a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
report={'status':'RUNNING','scope':'IMPORT_CONFIG_ONLY_NO_MODEL','imports':{}}
try:
 import torch
 torch.set_num_threads(2)
 import torch_npu
 torch.npu.set_device(0)
 for name in ('torch','torch_npu','kt_kernel','sglang.srt.layers.moe.kt_ep_wrapper',
              'sglang.srt.configs.qwen4_exp','sglang.srt.utils.hf_transformers_utils'):
  module=importlib.import_module(name);path=pathlib.Path(module.__file__)
  report['imports'][name]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
  print('IMPORTED',name,flush=True)
 from sglang.srt.layers.moe.kt_ep_wrapper import KTRANSFORMERS_AVAILABLE
 assert KTRANSFORMERS_AVAILABLE
 from transformers import AutoConfig
 cfg=AutoConfig.from_pretrained(a.model,local_files_only=True,trust_remote_code=False)
 c=cfg.text_config
 assert cfg.model_type=='qwen4_exp' and c.num_hidden_layers==48
 assert c.hidden_size==2560 and c.num_experts==512 and c.num_experts_per_tok==10
 assert c.hc_count==4 and c.hc_lowrank==320
 assert c.short_conv_layer_ids==[1] and c.short_conv_state_shape==(10240,9) and c.ngram_context_len==2
 assert [i for i,x in enumerate(c.layers_block_type) if x=='attention']==list(range(3,48,4))
 assert c.output_gate_type=='sigmoid' and c.rope_theta==10000000
 report['config']={'class':type(cfg).__name__,'text_class':type(c).__name__,
  'full_attention_layers':list(range(3,48,4)),'ple_layer':1,'short_conv_shape':[10240,9]}
 report['status']='PASS_IMPORT_CONFIG_ONLY'
except BaseException:
 report.update(status='FAIL',traceback=traceback.format_exc());raise
finally:a.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
