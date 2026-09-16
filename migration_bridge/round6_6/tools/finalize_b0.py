"""Close only backend decomposition; never claim model/quality qualification."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    base=Path('/workspace/results');checked={}
    def read(name):
        path=base/name;raw=path.read_bytes();checked[name]=hashlib.sha256(raw).hexdigest();return json.loads(raw)
    groups=[('r65-r66-threeway-v1','r65-r66-stages-v2',512),
            ('r65-r66-prefill-output-v2','r65-r66-prefill-stages-v2',96)]
    maxima={};observed=[]
    for prod,stage,count in groups:
        left=read(prod+'/three_way.json')['rows'];right=read(stage+'/three_way.json')['rows']
        assert len(left)==len(right)==count
        keys=lambda rs:[(r['layer'],r['expert'],r['call']) for r in rs]
        assert keys(left)==keys(right) and len(set(keys(left)))==count
        for x,y in zip(left,right):
            assert x['input_sha256']==y['input_sha256'] and x['weight_hashes']==y['weight_hashes']
            assert x['production_hashes']==y['production_hashes'] and x['npu_hashes']==y['npu_hashes']
            assert y['production_observer_parity']
            for r in [x,y]:
                assert r['finite'] and r['deterministic'] and r['local_safety_pass']
                assert len(r['production_hashes'])==len(r['npu_hashes'])==10
                assert len(set(r['production_hashes']))==len(set(r['npu_hashes']))==1
                for name in ['production_vs_npu','production_vs_canonical_fp32']:
                    m=r['comparisons'][name];assert m['finite'] and 0<=m['relative_l2']<=.01
                    maxima[name]=max(maxima.get(name,0),m['relative_l2'])
            for s,v in y['stage_comparisons'].items():
                for name,m in v.items():
                    assert m['finite'] and 0<=m['relative_l2']<=.01
                    key=s+'/'+name;maxima[key]=max(maxima.get(key,0),m['relative_l2'])
        observed+=left
    for name,count in [('r65-r66-layout-v1',512),('r65-r66-layout-prefill-v1',96)]:
        layout=read(name+'/layout.json');assert layout['status']=='NPU_ORACLE_ONE_VS_64_LAYOUT_EXACT'
        assert len(layout['rows'])==count and all(r['comparison']['exact'] for r in layout['rows'])
    before=read('r65-r66-runtime-source-v1.json');after=read('r65-r66-runtime-source-final-b0.json')
    assert before['files']==after['files'] and before['repositories']==after['repositories']
    gguf=base/'r65-r66-gguf-b0-guard-v1/process.log'
    assert gguf.read_text().split()[0]=='20114dba7790a9b6be40a17ffb3881c516b81ea86b8f2fcfd3ea61d052e96fe2'
    checked[str(gguf.relative_to(base))]=hashlib.sha256(gguf.read_bytes()).hexdigest()
    result=dict(status='BACKEND_DECOMPOSITION_GATE_PASS',expert_instances=128,operator_cases=608,
        decode_cases=384,prefill_cases=224,real_prefill_qlens=sorted({r['qlen'] for r in observed if r['qlen']>1}),
        maxima=maxima,evidence_sha256=checked,production_default_change=False,
        conclusion='No material additional LLAMAFILE numerical defect demonstrated in the frozen operator set.',
        limitations=['Unrouted real-hidden controls are not natural route-hit coverage.',
            'This is bounded sampled operator evidence, not a universal numerical proof.',
            'No model-level or downstream-quality qualification is implied.'],
        QD6='NOT_RUN',HD6='NOT_RUN',quality='NOT_RUN',ROUND6_5='BLOCKED_UNCHANGED',ROUND7='NOT_STARTED',
        next='FREEZE_INDEPENDENT_HEAVY_STATIC_CONTRACT_AND_CORPORA')
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
