"""Build the frozen Round10 T5 GDN/QSA/PLE component snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


MUTABLE={"round10-source-sha256.json","execution-budget.json","execution-budget.lock","host-controller.json"}


def _replace_once(source,old,new,label):
    if source.count(old)!=1:raise ValueError(f"{label} precondition mismatch")
    return source.replace(old,new)


def patch_gdn(source):
    source=_replace_once(source,"physical.value != 15","physical.value != 12","GDN physical")
    return _replace_once(source,'pathlib.Path("/dev/davinci15")','pathlib.Path("/dev/davinci12")',"GDN device")


def patch_qsa(source):
    source=_replace_once(source,"phy.value==15","phy.value==12","QSA physical")
    return _replace_once(source,"pathlib.Path('/dev/davinci15')","pathlib.Path('/dev/davinci12')","QSA device")


def patch_ple(source):
    source=_replace_once(source,"admission.get('physical')!=15","admission.get('physical')!=12","PLE admission")
    source=_replace_once(source,"physical.value!=15","physical.value!=12","PLE physical")
    source=_replace_once(source,"{Path('/dev/davinci15')}","{Path('/dev/davinci12')}","PLE device")
    source=_replace_once(source,"physical_device=15","physical_device=12","PLE receipt")
    old="source_sha256=source_sha,torch_version=str(torch.__version__),torch_npu_version=str(torch_npu.__version__))"
    new=old[:-1]+",expected_physical_device=12)"
    return _replace_once(source,old,new,"PLE gate expected physical")


def patch_ple_gate(source):
    source=_replace_once(source,"physical_device=15","physical_device=expected_physical_device",
                         "PLE gate physical")
    old=("def require_ple_device_gate(path, expected_sha, *, source_identity, source_sha256,\n"
         "                            torch_version, torch_npu_version):")
    new=("def require_ple_device_gate(path, expected_sha, *, source_identity, source_sha256,\n"
         "                            torch_version, torch_npu_version, expected_physical_device=15):")
    return _replace_once(source,old,new,"PLE gate signature")


def patch_guard(source):
    old="r'|kt-round10-t4-d-v[0-9]+)', name) is not None"
    new="r'|kt-round10-t4-d-v[0-9]+'\n        r'|kt-round10-t5-components-v[0-9]+)', name) is not None"
    return _replace_once(source,old,new,"T5 guard identity")


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest(root):
    return {str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and str(p.relative_to(root)) not in MUTABLE
            and not str(p.relative_to(root)).startswith('host-controller')}


def controller_source(version,training_id,prior=None):
    revision='de4b8e4d43b917e7706784d8bb445c9af86a3540'
    model=f'/models/{revision}'
    identity_receipt=f'{model}.round9-support/destination-verification/verification-1789886190714568317.json'
    if prior:
        prior_setup=f'''prior_spec={prior!r}
prior=pathlib.Path(prior_spec['root'])
supervisor=json.loads((prior/'component-supervisor.json').read_text())
if (supervisor.get('status')!='FAIL' or supervisor.get('owned_container_removed') is not True or
        supervisor.get('guard_exit_ack',{{}}).get('action') not in ('OWNED_EXITED_DURING_SAMPLE','OWNED_EXIT_CONFIRMED')):
 raise RuntimeError('prior T5 failure cleanup is not frozen')
for prior_name,prior_sha in (('gdn-result.json',prior_spec['gdn_sha256']),('qsa-result.json',prior_spec['qsa_sha256'])):
 prior_path=prior/prior_name
 if hashlib.sha256(prior_path.read_bytes()).hexdigest()!=prior_sha:raise RuntimeError('prior T5 success SHA changed')
print('PASS_PRIOR_T5_RESULTS_VERIFIED',flush=True)
'''
        prior_copy="shutil.copy2(prior/'gdn-result.json',out/'gdn-result.json');shutil.copy2(prior/'qsa-result.json',out/'qsa-result.json')"
        probe_script=(
          f'timeout -k 10 300 python -u /candidate/source/tests/ple_async_device_probe.py --model {model} --identity-receipt {identity_receipt} --admission /out/ple-admission.json --output /out/ple-result.json > /out/ple.log 2>&1; '
          'python -u /candidate/round10_t5_aggregate.py --gdn /out/gdn-result.json --qsa /out/qsa-result.json --ple /out/ple-result.json --output /out/result.json > /out/aggregate.log 2>&1')
    else:
        prior_setup=''
        prior_copy=''
        probe_script=(
          f'timeout -k 10 1200 python -u /candidate/source/tests/gdn_device_probe.py --model {model} --identity-receipt {identity_receipt} --output /out/gdn-result.json > /out/gdn.log 2>&1; '
          f'timeout -k 10 1200 python -u /candidate/source/tests/qsa_device_probe.py --model {model} --identity-receipt {identity_receipt} --output /out/qsa-result.json > /out/qsa.log 2>&1; '
          f'timeout -k 10 300 python -u /candidate/source/tests/ple_async_device_probe.py --model {model} --identity-receipt {identity_receipt} --admission /out/ple-admission.json --output /out/ple-result.json > /out/ple.log 2>&1; '
          'python -u /candidate/round10_t5_aggregate.py --gdn /out/gdn-result.json --qsa /out/qsa-result.json --ple /out/ple-result.json --output /out/result.json > /out/aggregate.log 2>&1')
    return f'''"""Run one guarded Round10 T5 combined component process."""
import fcntl,hashlib,json,pathlib,shutil,subprocess,time,traceback
from m3_cleanup import wait_for_guard_exit
root=pathlib.Path(__file__).resolve().parent;base=pathlib.Path('/mnt/data1/kt-round9')
out=base/'evidence'/'20260926-round10-t5-v{version}';name='kt-round10-t5-components-v{version}'
admission_path=pathlib.Path(__import__('sys').argv[1]);admission=json.loads(admission_path.read_text())
if admission.get('status')!='PASS' or time.time()-admission.get('ended_at',0)>600 or admission.get('training_id')!='{training_id}':raise RuntimeError('fresh T5 admission required')
source_manifest=json.loads((root/'round10-source-sha256.json').read_text());excluded={{'round10-source-sha256.json','execution-budget.json','execution-budget.lock','host-controller.json'}}
actual={{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and str(p.relative_to(root)) not in excluded and not str(p.relative_to(root)).startswith('host-controller')}}
if actual!=source_manifest:raise RuntimeError('T5 frozen snapshot changed')
{prior_setup}
out.mkdir(exist_ok=False);report={{'schema':1,'status':'PREPARING','version':{version},'started':time.time(),'container_name':name}}
{prior_copy}
claim=root/'host-controller.json'
with claim.open('x') as f:json.dump(report,f)
def save():claim.write_text(json.dumps(report,indent=2)+'\\n');(out/'component-supervisor.json').write_text(json.dumps(report,indent=2)+'\\n')
source=root/'source';ple_files=('ple_async.py','ple_rows.py','host_budget.py','framework_config.py')
ple_admission={{'status':'ADMITTED_PLE_COMPONENT_PENDING','physical':12,
 'source_sha256':{{n:hashlib.sha256((source/'runtime'/n).read_bytes()).hexdigest() for n in ple_files}},
 'probe_sha256':hashlib.sha256((source/'tests/ple_async_device_probe.py').read_bytes()).hexdigest(),
 'parent_admission_sha256':hashlib.sha256(admission_path.read_bytes()).hexdigest()}}
(out/'ple-admission.json').write_text(json.dumps(ple_admission,indent=2)+'\\n')
(out/'snapshot-contract.json').write_bytes((root/'ROUND10_T5_SNAPSHOT_CONTRACT.json').read_bytes())
rev='{revision}';receipt='{identity_receipt}'
script=('source /usr/local/Ascend/ascend-toolkit/set_env.sh; ulimit -c 0; set -e; '
 {probe_script!r})
cpus=admission['cpus'][:4]
cmd=['docker','create','--name',name,'--network','none','--cpuset-cpus',','.join(map(str,cpus)),'--cpuset-mems','2','--cpus','4','--memory','64g','--memory-swap','64g','--pids-limit','1024','--shm-size','4g']
for dev in ('davinci12','davinci_manager','devmm_svm','hisi_hdc'):cmd+=['--device','/dev/'+dev]
for mount in (f'{{root}}:/candidate:ro','/home/admin/models/Qwen3.8-Flash-Next:/models:ro',f'{{out}}:/out','/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro'):cmd+=['-v',mount]
for env in ('ASCEND_RT_VISIBLE_DEVICES=0','OMP_NUM_THREADS=2','OPENBLAS_NUM_THREADS=2','MKL_NUM_THREADS=2'):cmd+=['-e',env]
cmd+=['--entrypoint','bash','601e76d17e9f','-c',script]
container_id=None;guard=None
try:
 created=subprocess.check_output(cmd,text=True,timeout=30).strip();inspection=json.loads(subprocess.check_output(['docker','inspect',name],text=True,timeout=30))[0];container_id=inspection['Id']
 if created!=container_id or len(container_id)!=64:raise RuntimeError('T5 owned container identity mismatch')
 (out/'container-created.json').write_text(json.dumps(inspection,indent=2)+'\\n')
 with (out/'guard.log').open('w') as log:
  guard=subprocess.Popen(['python3',str(source/'tools/monitor_owned_server.py'),'--container',name,'--resource-mode','shared','--output',str(out/'monitor.jsonl'),'--ready-file',str(out/'guard-ready.json'),'--seconds','2800','--owned-limit-mib','11264','--training-container-id','{training_id}'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 deadline=time.monotonic()+40
 while not (out/'guard-ready.json').exists():
  if guard.poll() is not None or time.monotonic()>deadline:raise RuntimeError('T5 guard readiness failed')
  time.sleep(.2)
 with (root/'execution-budget.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);budget=json.loads((root/'execution-budget.json').read_text())
  if budget.get('stage')!='ROUND10_T5_COMPONENTS' or budget.get('component_process_used')!=0:raise RuntimeError('T5 component budget spent')
  budget['component_process_used']=1;budget['starts']=[{{'version':{version},'container_id':container_id,'time':time.time()}}]
  temp=root/'execution-budget.tmp';temp.write_text(json.dumps(budget,indent=2)+'\\n');temp.replace(root/'execution-budget.json')
 report.update(status='RUNNING',container_id=container_id,started_container=time.time(),command=cmd);save()
 subprocess.run(['docker','start',container_id],check=True,timeout=30,stdout=subprocess.DEVNULL)
 deadline=time.monotonic()+2700
 while time.monotonic()<deadline:
  state=json.loads(subprocess.check_output(['docker','inspect',name],text=True,timeout=30))[0]['State']
  if not state['Running']:
   report['container_exit']=state;break
  if any(p.stat().st_size>512*1024**2 for p in out.glob('*.log')):raise RuntimeError('T5 log ceiling exceeded')
  time.sleep(5)
 else:raise TimeoutError('T5 combined probe deadline')
 if state['ExitCode']!=0:raise RuntimeError('T5 component process failed; inspect component logs')
 result=json.loads((out/'result.json').read_text())
 if result.get('status')!='PASS_ROUND10_T5_COMPONENTS':raise RuntimeError('T5 combined component failed')
 report['status']='PASS';report['result_status']=result['status']
except BaseException as error:
 report.update(status='FAIL',error=repr(error),traceback=traceback.format_exc());raise
finally:
 report['ended']=time.time()
 try:
  if container_id:
   doc=json.loads(subprocess.check_output(['docker','inspect',name],text=True,timeout=30))[0]
   stop_started=time.time();was_running=doc['State']['Running']
   if was_running:subprocess.run(['docker','stop','-t','10',container_id],check=True,timeout=30)
   report['guard_exit_ack']=wait_for_guard_exit(out/'monitor.jsonl',not_before=stop_started if was_running else report.get('started_container',report['started']))
   subprocess.run(['docker','rm',container_id],check=True,timeout=30,stdout=subprocess.DEVNULL);report['owned_container_removed']=True
 except BaseException as cleanup:report['cleanup_error']=repr(cleanup);report['status']='FAIL_CLEANUP'
 save()
if report['status']!='PASS':raise SystemExit(1)
print('PASS_ROUND10_T5_COMPONENTS_AND_CLEANUP',flush=True)
'''


def build(args):
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    parent=args.parent
    shutil.copytree(parent/'source',args.output/'source',ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('m3_cleanup.py',):shutil.copy2(parent/name,args.output/name)
    tests=args.output/'source/tests'
    patches={'gdn_device_probe.py':patch_gdn,'qsa_device_probe.py':patch_qsa,'ple_async_device_probe.py':patch_ple}
    for name,function in patches.items():
        path=tests/name;path.write_text(function(path.read_text()));compile(path.read_text(),str(path),'exec')
    ple_gate=args.output/'source/runtime/ple_gate.py'
    ple_gate.write_text(patch_ple_gate(ple_gate.read_text()))
    compile(ple_gate.read_text(),str(ple_gate),'exec')
    guard=args.output/'source/tools/monitor_owned_server.py';guard.write_text(patch_guard(guard.read_text()))
    artifact=Path(__file__).resolve().parent
    shutil.copy2(artifact/'round10_t5_aggregate.py',args.output/'round10_t5_aggregate.py')
    prior=None
    if args.prior_evidence_root:
        if not args.prior_gdn_sha256 or not args.prior_qsa_sha256:
            raise ValueError('recovery requires both prior result SHA-256 values')
        prior={'root':args.prior_evidence_root,'gdn_sha256':args.prior_gdn_sha256,
               'qsa_sha256':args.prior_qsa_sha256}
    (args.output/'host_controller.py').write_text(controller_source(args.version,args.training_id,prior))
    budget={'schema':1,'stage':'ROUND10_T5_COMPONENTS','component_process_limit':1,'component_process_used':0,'starts':[],'automatic_retries_limit':0}
    (args.output/'execution-budget.json').write_text(json.dumps(budget,indent=2)+'\n')
    contract={'schema':1,'status':'FROZEN_NOT_RUN','version':args.version,'parent':'H0_ROUND10_V95_CURRENT_SOURCE_V102',
              'components':['GDN_REAL_WEIGHT_INDEPENDENT_FP64','QSA_REAL_WEIGHT_INDEX_AND_ATTENTION_FP64','PLE_ASYNC_ORIGINAL_ROWS_OWNERSHIP'],
              'physical_device':12,'component_process_limit':1,'automatic_retries':0}
    if prior:
        contract['recovery']={'scope':'PLE_ONLY','reused_results':prior}
    (args.output/'ROUND10_T5_SNAPSHOT_CONTRACT.json').write_text(json.dumps(contract,indent=2)+'\n')
    files=manifest(args.output);(args.output/'round10-source-sha256.json').write_text(json.dumps(files,indent=2,sort_keys=True)+'\n')
    print('PASS_ROUND10_T5_SNAPSHOT',args.version,len(files),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--parent',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--version',type=int,default=103);parser.add_argument('--training-id',required=True)
    parser.add_argument('--prior-evidence-root');parser.add_argument('--prior-gdn-sha256');parser.add_argument('--prior-qsa-sha256')
    build(parser.parse_args())


if __name__=='__main__':main()
