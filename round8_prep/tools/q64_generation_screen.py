"""32 distinct prompts, repeated short generations; no benchmark quality score."""
import hashlib
import argparse
import json
import math
from pathlib import Path
import requests
from transformers import AutoTokenizer

PROMPTS=[
 '用一句话解释光合作用。','为什么水在高山上更容易沸腾？',
 '描述图书馆的两个用途。','给初学者解释什么是质数。',
 '写一句包含春天和微风的句子。','简述公共交通的一个优点。',
 '列出两种可再生能源。','说明冰和液态水的一个区别。',
 'Explain what a compiler does in one sentence.','Name two differences between TCP and UDP.',
 'What does a dictionary store in Python?','Describe one use of a queue.',
 'Explain why the Moon has phases.','Give two examples of mammals.',
 'What is the purpose of a unit test?','Describe a safe way to back up files.',
 'Calculate 12 + 37.','Calculate 9 times 14.','What is half of 86?',
 'Convert 2.5 hours to minutes.','Find the next number: 3, 6, 9, 12.',
 'Solve x + 7 = 19.','What is the area of a rectangle 4 by 7?',
 'Compare 3/4 and 2/3. Which is larger?',
 'Return JSON with key ok and boolean value true.',
 'Return a JSON array containing red and blue.',
 'Write a Python function that adds two numbers.',
 'Write one SQL statement selecting all rows from books.',
 'Explain the difference between == and = in Python.',
 'Return only a comma-separated list of the numbers 1, 2, 3.',
 '给出一个只含 name 字段且值为小明的 JSON 对象。',
 '用一行 Python 代码计算列表 [1, 2, 3] 的和。']

def main():
 p=argparse.ArgumentParser();p.add_argument('--profile',choices=['q64','q32'],default='q64');a=p.parse_args()
 root=Path('/workspace/round8');out=root/f'{a.profile}-generation-screen-v1'
 assert json.loads((root/f'{a.profile}-compact-v1/restart-summary.json').read_text())['status']=='PASS'
 candidate='q64-launch-v6' if a.profile=='q64' else 'q32-launch-restart-v1'
 structure=[json.loads(s) for s in (root/candidate/'structure.jsonl').read_text().splitlines()]
 assert {r['layer'] for r in structure if r.get('shared_exactly_once_oracle')}==set(range(40))
 assert all(r['slots']==int(a.profile[1:]) for r in structure)
 out.mkdir(exist_ok=False)
 tokenizer=AutoTokenizer.from_pretrained(root/'model-sample',local_files_only=True)
 frozen=[]
 for i,text in enumerate(PROMPTS):
  rendered=tokenizer.apply_chat_template([dict(role='user',content=text)],tokenize=False,add_generation_prompt=True,enable_thinking=False)
  frozen.append(dict(id=f'compact_{i:02}',text=text,input_ids=tokenizer.encode(rendered,add_special_tokens=False)))
 protocol=dict(prompts=frozen,repeats=2,max_new_tokens=8,temperature=0,thinking=False,
  purpose='SYSTEM_GENERATION_SCREEN_NOT_CMMLU_OR_PERFORMANCE')
 blob=json.dumps(protocol,ensure_ascii=False,sort_keys=True,indent=2)+'\n'
 (out/'protocol.json').write_text(blob)
 with (out/'raw.jsonl').open('x') as f:
  for row in frozen:
   first=None
   for repeat in range(2):
    response=requests.post('http://127.0.0.1:18400/generate',json=dict(input_ids=row['input_ids'],
      sampling_params=dict(temperature=0,max_new_tokens=8),return_logprob=True),timeout=900)
    data=response.json();f.write(json.dumps(dict(id=row['id'],repeat=repeat,status=response.status_code,response=data),ensure_ascii=False)+'\n');f.flush()
    response.raise_for_status();assert data['text'].strip() and data['output_ids']
    assert all(math.isfinite(x[0]) for x in data['meta_info']['output_token_logprobs'])
    if first is None:first=data['output_ids']
    else:assert first==data['output_ids'], 'NONDETERMINISTIC:'+row['id']
   print(row['id']+' PASS',flush=True)
 (out/'summary.json').write_text(json.dumps(dict(status='PASS',prompts=32,requests=64,
  protocol_sha256=hashlib.sha256(blob.encode()).hexdigest(),quality_qualified=False,performance_qualified=False),indent=2)+'\n')

if __name__=='__main__':main()
