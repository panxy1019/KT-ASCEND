"""Freeze non-quality prefill shapes before observing C32/C64 performance."""
import json
from pathlib import Path
from transformers import AutoTokenizer
from common import sha, write

root=Path('/workspace/c64-dev/performance-freeze-v1');root.mkdir(exist_ok=False)
tokenizer=Path('/workspace/results/r64-tokenizer-compat-v1')
t=AutoTokenizer.from_pretrained(tokenizer,local_files_only=True,trust_remote_code=True)
text=('Describe how a computer stores and retrieves information. Explain memory hierarchy, '
      'data movement, reproducibility, and the distinction between capacity and latency. ')
unit=t.encode(text,add_special_tokens=False)
assert len(unit)>0
rows=[dict(id=f'prefill-{n}',input_ids=(unit*((n+len(unit)-1)//len(unit)))[:n],max_new_tokens=1)
      for n in (256,384,512,640,768)]
report=dict(version='c64-prefill-v1',rows=rows,blocks=3,order=['C32','C64','C64','C32'],
    warmup_per_process=1,greedy=True,batch=1,radix_cache=False,observer=False,
    primary='median_paired_wall_latency',beneficial_ratio_max=.85,marginal_ratio_max=.90,
    formal_requires_host_bandwidth_isolation=True,
    tokenizer_files={p.name:sha(p) for p in sorted(tokenizer.iterdir()) if p.is_file()})
write(root/'perf_prompt_set.json',report)
print(json.dumps(dict(path=str(root/'perf_prompt_set.json'),sha256=sha(root/'perf_prompt_set.json'))))
