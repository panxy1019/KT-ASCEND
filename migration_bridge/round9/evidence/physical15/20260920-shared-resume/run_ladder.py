import subprocess as s
import json,time,pathlib
p=pathlib.Path('/mnt/data1/kt-round9/evidence/20260920-shared-resume')
image='sha256:601e76d17e9fb4b1c5a72d491cf086c533a309c5c651315e73fe783c90460c8a'
rows=[]
for mib in (256,1024,2048):
    before=s.check_output(['npu-smi','info'],text=True)
    (p/f'before-{mib}.txt').write_text(before)
    usage=s.check_output(['npu-smi','info','-t','usages','-i','7','-c','1'],text=True)
    (p/f'usage-{mib}.txt').write_text(usage)
    percent=int(next(x.split(':')[1] for x in usage.splitlines() if 'HBM Usage Rate' in x))
    # Integer telemetry rounded upward; 4 GiB extra runtime budget and 12 GiB reserve.
    assert (percent+1)*65536/100+mib+4096+12288<=65536
    name=f'kt-round9-phy15-{mib}'
    cmd=['docker','run','--name',name,'--network','none','--memory','4g','--memory-swap','4g','--cpus','2','--pids-limit','512']
    for d in ('davinci15','davinci_manager','devmm_svm','hisi_hdc'): cmd+=['--device','/dev/'+d]
    cmd+=['-v','/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro','-v',str(p)+':/probe:ro','-e','ASCEND_RT_VISIBLE_DEVICES=0','--entrypoint','/bin/bash',image,'-c',f'source /usr/local/Ascend/ascend-toolkit/set_env.sh; timeout 90 python /probe/shared_probe_v2.py {mib}']
    (p/f'command-{mib}.json').write_text(json.dumps(cmd,indent=2))
    with (p/f'probe-{mib}.log').open('w') as log, (p/f'monitor-{mib}.jsonl').open('w') as mon:
        proc=s.Popen(cmd,stdout=log,stderr=s.STDOUT)
        while proc.poll() is None:
            snapshot=s.run(['npu-smi','info'],capture_output=True,text=True,timeout=10)
            mon.write(json.dumps({'time':time.time(),'rc':snapshot.returncode,'output':snapshot.stdout})+'\n');mon.flush()
            time.sleep(2)
    inspect=json.loads(s.check_output(['docker','inspect',name],text=True))[0]
    (p/f'container-{mib}.json').write_text(json.dumps({'Id':inspect['Id'],'State':inspect['State'],'HostConfig':{k:inspect['HostConfig'][k] for k in ('Devices','Memory','MemorySwap','NanoCpus','Privileged','NetworkMode')}},indent=2))
    after=s.check_output(['npu-smi','info'],text=True);(p/f'after-{mib}.txt').write_text(after)
    passed=proc.returncode==0 and 'PASS_BOUNDED_RUNTIME_ONLY' in (p/f'probe-{mib}.log').read_text()
    rows.append({'mib':mib,'exit_code':proc.returncode,'passed':passed})
    (p/'ladder-results.json').write_text(json.dumps(rows,indent=2))
    print(json.dumps(rows[-1]),flush=True)
    if not passed: break
