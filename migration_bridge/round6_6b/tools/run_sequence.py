"""One All-NPU/Hybrid sequence, fixed inputs, stop on infrastructure failure."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys


INPUT_SHA='885ccf76386dcfed2867142417300ac69c4401754360f92e6b6e4f179adffd97'


def main():
    import numpy as np
    base=Path('/workspace/results');out=base/'r65-r66b-base-protocol-v1'
    tools=Path(__file__).parent
    inputs=base/'r65-r66b-base-protocol-freeze-v1/quality.json'
    assert hashlib.sha256(inputs.read_bytes()).hexdigest()==INPUT_SHA
    out.mkdir(exist_ok=False)
    status=dict(status='RUNNING',completed=[],input_sha256=INPUT_SHA,
                population_noninferiority_claim=False,old_v1_status='BLOCKED_UNCHANGED')
    def save():(out/'progress.json').write_text(json.dumps(status,indent=2)+'\n')
    try:
        for mode,profile in [('npu','four-R64'),('hybrid','four-R32')]:
            status['status']='RUNNING_'+mode.upper();save()
            with (out/(mode+'.log')).open('x') as log:
                subprocess.run([sys.executable,str(tools/'run_base_protocol.py'),
                    '--inputs',str(inputs),'--input-sha256',INPUT_SHA,
                    '--profile',str(base/'r65-profiles-v1'/(profile+'.json')),
                    '--output',str(out/mode)],stdout=log,stderr=subprocess.STDOUT,check=True)
            status['completed'].append(mode);save()
        n=json.loads((out/'npu/quality-report.json').read_text())
        h=json.loads((out/'hybrid/quality-report.json').read_text())
        assert n['status']==h['status']=='QUALITY_SCREEN_COLLECTION_COMPLETE'
        assert [r['id'] for r in n['rows']]==[r['id'] for r in h['rows']]
        assert n['input_sha256']==h['input_sha256']==INPUT_SHA
        reports=[]
        for bench,size in [('gsm8k',32),('cmmlu',256)]:
            pairs=[(x,y) for x,y in zip(n['rows'],h['rows']) if x['benchmark']==bench]
            assert len(pairs)==size and all(y['benchmark']==bench for x,y in pairs)
            gain=sum(not x['correct'] and y['correct'] for x,y in pairs)
            loss=sum(x['correct'] and not y['correct'] for x,y in pairs)
            ni=sum(x['invalid'] for x,y in pairs);hi=sum(y['invalid'] for x,y in pairs)
            rng=np.random.default_rng(660602)
            b=rng.multinomial(size,[gain/size,loss/size,1-(gain+loss)/size],size=20000)
            interval=np.quantile((b[:,0]-b[:,1])/size,[.025,.975]).tolist()
            reports.append(dict(benchmark=bench,n=size,
                correct_npu=sum(x['correct'] for x,y in pairs),correct_hybrid=sum(y['correct'] for x,y in pairs),
                gains=gain,losses=loss,invalid_npu=ni,invalid_hybrid=hi,
                truncated_npu=sum(x['truncated'] for x,y in pairs),truncated_hybrid=sum(y['truncated'] for x,y in pairs),
                bootstrap_percentile95=interval,screen_pass=gain>=loss and hi<=ni))
        status.update(status='SCREEN_PASS_AWAIT_AUDIT' if all(r['screen_pass'] for r in reports)
                      else 'SCREEN_FAILED_NO_RESEED',benchmarks=reports)
    except BaseException as exc:
        status.update(status='INFRASTRUCTURE_FAILURE',error=repr(exc));raise
    finally:save()


if __name__=='__main__':main()
