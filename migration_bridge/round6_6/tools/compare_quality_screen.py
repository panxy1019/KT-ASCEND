"""Frozen zero-observed-loss screen; intervals do not imply population equivalence."""
import argparse
import json
from pathlib import Path
import sys


def main():
    import numpy as np
    p=argparse.ArgumentParser()
    p.add_argument('--npu',type=Path,required=True)
    p.add_argument('--hybrid',type=Path,required=True)
    p.add_argument('--statistics-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();sys.path.insert(0,str(a.statistics_dir))
    from paired_quality_interval import bounds
    n=json.loads(a.npu.read_text());h=json.loads(a.hybrid.read_text())
    assert n['status']==h['status']=='QUALITY_SCREEN_COLLECTION_COMPLETE'
    assert n['input_sha256']==h['input_sha256']=='9618d054cf765e1d6328d1581435ec97f9f4271715b794dc07e0a9889a483fcb'
    assert n['approval_sha256']==h['approval_sha256']
    assert n['profile_sha256']!=h['profile_sha256']
    assert [r['id'] for r in n['rows']]==[r['id'] for r in h['rows']]
    assert len(n['rows'])==len(h['rows'])==288
    reports=[]
    for bench,size in [('gsm8k',32),('cmmlu',256)]:
        pairs=[(x,y) for x,y in zip(n['rows'],h['rows']) if x['benchmark']==bench]
        assert len(pairs)==size and all(y['benchmark']==bench for x,y in pairs)
        gain=sum(not x['correct'] and y['correct'] for x,y in pairs)
        loss=sum(x['correct'] and not y['correct'] for x,y in pairs)
        ni=sum(x['invalid'] for x,y in pairs);hi=sum(y['invalid'] for x,y in pairs)
        rng=np.random.default_rng(6606)
        counts=rng.multinomial(size,[gain/size,loss/size,1-(gain+loss)/size],size=20000)
        interval=np.quantile((counts[:,0]-counts[:,1])/size,[.025,.975]).tolist()
        reports.append(dict(benchmark=bench,n=size,correct_npu=sum(x['correct'] for x,y in pairs),
            correct_hybrid=sum(y['correct'] for x,y in pairs),invalid_npu=ni,invalid_hybrid=hi,
            truncated_npu=sum(x['truncated'] for x,y in pairs),truncated_hybrid=sum(y['truncated'] for x,y in pairs),
            gains=gain,losses=loss,exact_marginal_interval=bounds(size,gain,loss),
            bootstrap_percentile95=interval,bootstrap_seed=6606,bootstrap_replicates=20000,
            observed_screen_pass=gain>=loss and hi<=ni))
    result=dict(status='PAIRED_SCREEN_PASS_AWAIT_ADR' if all(r['observed_screen_pass'] for r in reports)
                else 'PAIRED_SCREEN_FAILED',benchmarks=reports,population_noninferiority_claim=False,
                caveat='Subset collapse, reference floor and truncation require explicit ADR review; bootstrap zero width is not zero population uncertainty')
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
