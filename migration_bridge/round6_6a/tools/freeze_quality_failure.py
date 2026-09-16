"""Freeze every quality correctness discordance, including improvements."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--quality',type=Path,required=True)
    p.add_argument('--pairs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    assert sha(a.quality)=='9618d054cf765e1d6328d1581435ec97f9f4271715b794dc07e0a9889a483fcb'
    assert sha(a.pairs)=='d0b839b7cdaa162cea321242de330c06ed63a9ac9b9fb42f316c76714e8e01da'
    pairs=json.loads(a.pairs.read_text())['rows']
    quality={r['id']:r for r in json.loads(a.quality.read_text())['rows']}
    changed=[r for r in pairs if r['state'] in ('gain','loss')]
    assert len(changed)==8 and sum(r['state']=='loss' for r in changed)==6
    requests=[]
    for i,row in enumerate(changed):
        source=quality[row['id']]
        assert 0<=row['first_token_difference']<12
        requests.append(dict(rid=f'r66a-discordance-{i:02d}',quality_id=row['id'],
            family=row['benchmark'],input_ids=source['input_ids'],state=row['state'],
            original_first_difference=row['first_token_difference'],
            original_npu_tokens=row['npu']['tokens'],original_hybrid_tokens=row['hybrid']['tokens'],
            post_original_stop='DIAGNOSTIC_ONLY_NOT_QUALITY_REPRODUCTION'))
    result=dict(version='r66a-all-quality-discordances-v1',scope='DIAGNOSTIC_NOT_HELDOUT',
                token_budget=12,quality_sha256=sha(a.quality),pairs_sha256=sha(a.pairs),requests=requests)
    with a.output.open('x') as f:
        json.dump(result,f,indent=2);f.write('\n')
    print(sha(a.output))


if __name__=='__main__':main()
