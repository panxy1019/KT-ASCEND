"""Select first four Q2 and H2 prompts before any heavy-profile results."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--corpora',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    rows=[]; hashes={}
    for name in ('q2','h2'):
        raw=(a.corpora/(name+'.json')).read_bytes()
        hashes[name]=hashlib.sha256(raw).hexdigest()
        doc=json.loads(raw)
        for r in doc['prompts'][:4]:
            rows.append(dict(rid='r65-'+name+'-'+r['id'],input_ids=r['input_ids'],
                             prompt_id=r['id'],corpus=name,text=r['text']))
    obj=dict(version='r65-static-numerical-compact-v1',token_budget=64,requests=rows,
             source_sha256=hashes,selection='first_four_each_in_frozen_order',B_pair=2.1875,K=32)
    with a.output.open('x') as f: json.dump(obj,f,indent=2,ensure_ascii=False);f.write('\n')


if __name__=='__main__':main()
