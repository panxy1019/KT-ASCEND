import hashlib,json
from pathlib import Path
from transformers import AutoTokenizer
model='/models/de4b8e4d43b917e7706784d8bb445c9af86a3540'
prompts=[
 {'id':'sched-dev-01','text':'Explain in two short sentences how a queue differs from a stack.'},
 {'id':'sched-dev-02','text':'用两句话解释植物为什么需要阳光。'},
]
tok=AutoTokenizer.from_pretrained(model,local_files_only=True,trust_remote_code=True)
rows=[]
for row in prompts:
 ids=tok.apply_chat_template([{'role':'user','content':row['text']}],tokenize=True,add_generation_prompt=True,enable_thinking=False)
 rows.append({**row,'input_ids':[int(x) for x in ids],'input_tokens':len(ids),'input_ids_sha256':hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()})
out={'schema':1,'status':'FROZEN_ROUND13_DEV_TOKEN_IDS','split':'QWEN38_SCHED_DEV','model_revision':'de4b8e4d43b917e7706784d8bb445c9af86a3540','chat_template':'ACTUAL_TOKENIZER_ENABLE_THINKING_FALSE','prompts':rows}
Path('/out/round13-dev-fixture.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':out['status'],'lengths':[x['input_tokens'] for x in rows]},ensure_ascii=False))
