"""Create two isolated .22 test lanes; no model launch or performance claims."""
import json
import subprocess
from pathlib import Path

ROOT=Path('/home/admin/kt-round8-qwen35')
def cmd(*args):
    return subprocess.check_output(args,text=True,timeout=180)

def main():
    state=json.loads((ROOT/'qwen-static-perf-v6/controller-state.json').read_text())
    assert state['status']=='COLLECTION_COMPLETE_PENDING_RESOURCE_AUDIT'
    base=json.loads(cmd('docker','inspect','kt-round8-qwen35-prep'))[0]
    records=[]
    for lane,npu,cpus,mems,old in (
        ('a',4,'0-63','0,1','kt-r7-parallel-a-v1'),
        ('b',6,'64-127','2,3','kt-r7-parallel-b-v1')):
        top=cmd('docker','top',old).splitlines()
        assert len(top)==2 and '/bin/sleep infinity' in top[1],top
        subprocess.run(['docker','stop','-t','20',old],check=True,timeout=60)
        name=f'kt-round8-qwen35-dynamic-{lane}'
        assert subprocess.run(['docker','inspect',name],capture_output=True).returncode!=0
        args=['docker','create','--name',name,'--cpuset-cpus',cpus,
              '--cpuset-mems',mems,'--memory','384g','--memory-swap','384g',
              '--shm-size','16g','--network','none','--cap-add','SYS_NICE']
        for device in (f'davinci{npu}','davinci_manager','devmm_svm','hisi_hdc'):
            args+=['--device',f'/dev/{device}']
        for mount in base['HostConfig']['Binds']:
            args+=['-v',mount]
        args+=['-e','ASCEND_RT_VISIBLE_DEVICES=0',base['Image'],'infinity']
        cmd(*args);cmd('docker','start',name)
        probe=cmd('docker','exec',name,'/workspace/round8/venv/bin/python','-c',
            'import torch,torch_npu; assert torch.npu.device_count()==1; '
            'x=torch.arange(16,dtype=torch.float32).npu(); '
            'assert torch.equal(x.cpu(),torch.arange(16,dtype=torch.float32)); '
            'torch.npu.synchronize(); print("DEVICE_COPY_PASS")')
        records.append(dict(container=name,npu=npu,cpus=cpus,mems=mems,
                            probe=probe,c64='2x32',timing_qualified=False))
    (ROOT/'parallel22-lanes.json').write_text(json.dumps(records,indent=2)+'\n')
    print(json.dumps(records,indent=2))

if __name__=='__main__':main()
