"""One isolated-NPU quality lane; parallel timing is explicitly non-formal."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

ROOT=Path('/home/admin/kt-round8-qwen35');PY='/workspace/round8/venv/bin/python'

def command(*args,timeout=60):return subprocess.check_output(args,text=True,timeout=timeout)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['heldout','cmmlu'],required=True)
    p.add_argument('--serve',action='store_true');a=p.parse_args()
    name=f'qwen-dynamic-{a.mode}-v2'
    if a.serve:
        root=Path('/workspace/round8');out=root/name
        cfg=json.loads((out/'resolved-command.json').read_text())
        for path,digest in cfg['code_hashes'].items():assert sha(Path(path))==digest,path
        env=os.environ.copy();env.update(cfg['env']);os.execvpe(cfg['argv'][0],cfg['argv'],env)
    from round8_resource_guard import reject_reasons
    frozen=ROOT/'qwen-dynamic-candidate-v2'
    ack=json.loads((frozen/'freeze-ack.json').read_text())
    contract=json.loads((frozen/'candidate.json').read_text())
    for tool,digest in contract['tools_sha256'].items():assert sha(ROOT/'tools'/tool)==digest,tool
    assert ack['candidate_sha256']==sha(frozen/'candidate.json')
    assert len(ack['local_commit'])==40 and ack['system_gate']=='PASS'
    source=frozen/(a.mode+'.json');assert sha(source)==contract['configs_sha256'][a.mode]
    out=ROOT/name;out.mkdir(exist_ok=False)
    (out/'entry.json').write_text(json.dumps(ack,indent=2));shutil.copyfile(source,out/'resolved-command.json')
    npu,cpus,nodes,lane=(4,'0-63','0,1','a') if a.mode=='heldout' else (6,'64-127','2,3','b')
    container=f'kt-round8-qwen35-dynamic-{lane}';started=False
    state=dict(status='ADMISSION',mode=a.mode,container=container,formal_performance=False)
    def save():(out/'state.json').write_text(json.dumps(state,indent=2))
    save()
    try:
        info=json.loads(command('docker','inspect',container))[0]
        assert not info['State']['Running'],'do not interrupt healthy lane'
        hc=info['HostConfig'];assert hc['CpusetCpus']==cpus and hc['CpusetMems']==nodes
        assert hc['Memory']==hc['MemorySwap']==384*2**30
        ids=[int(re.fullmatch(r'/dev/davinci(\d+)',d['PathOnHost'])[1]) for d in hc['Devices'] if re.fullmatch(r'/dev/davinci(\d+)',d['PathOnHost'])]
        assert ids==[npu]
        command('docker','start',container);started=True
        def vm():return dict((k,int(v)) for k,v in (s.split() for s in Path('/proc/vmstat').read_text().splitlines()))
        before=vm();t=time.monotonic();time.sleep(30);after=vm()
        mem={s.split(':')[0]:int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines()}
        proc=command('npu-smi','info','-t','proc-mem','-i',str(npu),'-c','0')
        usage=command('npu-smi','info','-t','usages','-i',str(npu),'-c','0')
        capacity=int(re.search(r'HBM Capacity\(MB\)\s*:\s*(\d+)',usage)[1])*2**20
        percent=int(re.search(r'HBM Usage Rate\(%\)\s*:\s*(\d+)',usage)[1])
        topology=[s.split(',') for s in command('lscpu','-p=CPU,CORE,SOCKET,NODE').splitlines() if not s.startswith('#')]
        lo,hi=map(int,cpus.split('-'));physical=[r for r in topology if lo<=int(r[0])<=hi]
        assert {r[3] for r in physical}==set(nodes.split(','))
        inventory=dict(inventory_age_seconds=0,round7_perf_active=False,round7_npus=[],round7_cpus=[],
            requested_npu=npu,requested_cpus=list(range(lo,hi+1)),host_available_bytes=mem['MemAvailable'],
            host_total_bytes=mem['MemTotal'],predicted_private_bytes=256*2**30,
            swap_used_bytes=mem['SwapTotal']-mem['SwapFree'],npu_free_bytes=int(capacity*(99-percent)/100),
            predicted_npu_peak_bytes=44*2**30,disk_free_bytes=shutil.disk_usage(ROOT).free,
            download_conversion_budget_bytes=20*2**30,other_npu_processes=[] if 'No process in device.' in proc else [proc],
            same_host_inventory=True,physical_core_count=len({(r[1],r[2]) for r in physical}),
            host_swapin_delta=after['pswpin']-before['pswpin'],host_swapout_delta=after['pswpout']-before['pswpout'],
            host_page_size_bytes=os.sysconf('SC_PAGE_SIZE'),swap_observation_seconds=time.monotonic()-t,
            cgroup_swap_current=int(command('docker','exec',container,'cat','/sys/fs/cgroup/memory.swap.current')))
        reasons=reject_reasons(inventory)
        (out/'admission.json').write_text(json.dumps(dict(inventory=inventory,reasons=reasons,physical_topology=physical),indent=2))
        assert not reasons,reasons
        command('docker','exec','-d',container,'bash','-c',
            f'{PY} /workspace/round8/tools/run_qwen_dynamic_evaluation.py --mode {a.mode} --serve > /workspace/round8/{name}/server.log 2>&1')
        state['status']='LOADING';save();deadline=time.monotonic()+1200
        while True:
            r=subprocess.run(['docker','exec',container,PY,'-c','import requests; assert requests.get("http://127.0.0.1:18400/health",timeout=3).status_code==200'],capture_output=True,timeout=15)
            if r.returncode==0:break
            assert time.monotonic()<deadline,'readiness timeout';time.sleep(5)
        state['status']='RUNNING';save()
        with (out/'client.log').open('x') as log, (out/'resources.jsonl').open('x') as resources:
            client=subprocess.Popen(['docker','exec',container,PY,'/workspace/round8/tools/qwen_dynamic_evaluation.py',
                '--mode',a.mode,'--output',f'/workspace/round8/{name}'],stdout=log,stderr=subprocess.STDOUT)
            while client.poll() is None:
                info=json.loads(command('docker','inspect',container))[0]
                swap=int(command('docker','exec',container,'cat','/sys/fs/cgroup/memory.swap.current'))
                events=command('docker','exec',container,'cat','/sys/fs/cgroup/memory.events')
                row=dict(timestamp=time.time(),state=info['State'],swap=swap,memory_events=events,
                    stats=command('docker','stats','--no-stream','--format','{{json .}}',container),npu=command('npu-smi','info'))
                resources.write(json.dumps(row)+'\n');resources.flush()
                assert not info['State']['OOMKilled'] and swap==0,'OWNED_RESOURCE_FAILURE'
                assert not any(int(line.split()[1]) for line in events.splitlines() if line.split()[0] in ('oom','oom_kill'))
                assert 'Check input storage_offset' not in (out/'server.log').read_text(),'TRANSFER_WARNING_REAPPEARED'
                time.sleep(30)
            assert client.wait()==0,'evaluation client failed'
        state['status']='COLLECTED_PENDING_INDEPENDENT_AUDIT'
    except BaseException as error:
        state.update(status='FAILED',error=repr(error));raise
    finally:
        save()
        if started:subprocess.run(['docker','stop','-t','60',container],capture_output=True,timeout=150)

if __name__=='__main__':main()
