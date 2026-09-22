"""One short diagnostic request, never a quality/performance benchmark."""
import json
from pathlib import Path
import requests
from transformers import AutoTokenizer


def main():
    root=Path('/workspace/round8')
    t=AutoTokenizer.from_pretrained(root/'model-sample',local_files_only=True)
    rendered=t.apply_chat_template([{'role':'user','content':'用一句话介绍你自己。'}],
        tokenize=False,add_generation_prompt=True,enable_thinking=False)
    ids=t.encode(rendered,add_special_tokens=False)
    assert isinstance(ids,list) and len(ids)>2 and all(isinstance(i,int) for i in ids)
    payload=dict(input_ids=ids,sampling_params=dict(temperature=0,max_new_tokens=8),
                 return_logprob=True)
    out=root/'q64-launch-v5'
    (out/'smoke-request.json').write_text(json.dumps(dict(rendered=rendered,request=payload),ensure_ascii=False,indent=2)+'\n')
    response=requests.post('http://127.0.0.1:18400/generate',json=payload,timeout=900)
    (out/'smoke-response.txt').write_text(response.text)
    response.raise_for_status()
    data=response.json()
    assert isinstance(data['text'],str) and data['text'].strip(), 'EMPTY_GENERATION'
    assert data['meta_info']['completion_tokens']>0
    structure=[json.loads(l) for l in (out/'structure.jsonl').read_text().splitlines()]
    assert {r['layer'] for r in structure}==set(range(40)), 'INCOMPLETE_40_LAYER_AUDIT'
    routes=[json.loads(l) for l in (out/'routing.jsonl').read_text().splitlines()]
    assert {r['layer'] for r in routes}==set(range(40))
    result=dict(status='SHORT_Q64_BRINGUP_PASS_NOT_QUALIFICATION',tokens=data['meta_info']['completion_tokens'],
        structure_layers=40,cpu_hits=sum(r['cpu_hits'] for r in routes),
        npu_hits=sum(r['npu_hits'] for r in routes),text=data['text'])
    (out/'smoke-summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
