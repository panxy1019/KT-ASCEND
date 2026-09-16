"""Independent operator-only prefill corpus; never QD/HD or quality data."""
import argparse
import hashlib
import json
from pathlib import Path
from transformers import AutoTokenizer


def main():
    p=argparse.ArgumentParser();p.add_argument('--tokenizer',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    tokenizer=AutoTokenizer.from_pretrained(a.tokenizer,local_files_only=True,trust_remote_code=False)
    text=('A laboratory notebook records measurements of water temperature, clock time, and container mass. '
          'Describe how to keep the observations organized without inventing missing measurements. ')*20
    ids=tokenizer.encode(text,add_special_tokens=False)
    rows=[]
    for count in [16,64,128]:
        sample=ids[:count]
        assert len(sample)==count
        rows.append(dict(rid=f'r66-op-prefill-{count}',input_ids=sample,
                         prompt_text=tokenizer.decode(sample),category='OPERATOR_ONLY_REAL_PREFILL'))
    data=dict(version='r66-operator-prefill-v1',token_budget=1,requests=rows,
              purpose='LOCAL_OPERATOR_OPERANDS_NOT_MODEL_THRESHOLD_FITTING',generation_seed=0,
              selected_cpu_experts='all odd IDs in layers 1,9,17,26; unrouted controls explicitly labelled')
    raw=(json.dumps(data,ensure_ascii=False,indent=2)+'\n').encode()
    with a.output.open('xb') as f:f.write(raw)
    print(hashlib.sha256(raw).hexdigest())


if __name__=='__main__':main()
