"""Create one bounded owned container, arm its guard, refresh admission, start.

No scheduler/automation is created. All logs and failed containers are retained.
"""
import argparse,json,pathlib,subprocess,time
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--version',type=int,required=True);a=p.parse_args()
assert a.version>0
root=pathlib.Path('/mnt/data1/kt-round9');base=root/'src/KT-ASCEND/migration_bridge/round9'
name=f'kt-round9-full-model-v{a.version}'
out=root/'evidence'/f'20260920-full-model-v{a.version}';out.mkdir(exist_ok=False)
container_script='source /usr/local/Ascend/ascend-toolkit/set_env.sh; exec timeout -k 10 3600 python -u /round9/src/KT-ASCEND/migration_bridge/round9/runtime/full_model_server.py --profile /round9/src/KT-ASCEND/migration_bridge/round9/runtime/profiles/q64_bf16_c64_sync_physical15.json --resolved-profile /out/resolved-profile.json --admission /out/preflight/admission.json > /out/server.log 2>&1'
args=['docker','create','--name',name,'--network','none','--cpuset-cpus','480-639','--cpus','72','--memory','640g','--memory-swap','640g','--pids-limit','4096','--shm-size','16g','--cap-add','SYS_NICE',
 '--device','/dev/davinci15','--device','/dev/davinci_manager','--device','/dev/devmm_svm','--device','/dev/hisi_hdc',
 '-v',f'{root}:/round9:ro','-v','/home/admin/models/Qwen3.8-Flash-Next:/models:ro','-v',f'{out}:/out','-v','/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro',
 '-e','ASCEND_RT_VISIBLE_DEVICES=0','-e','OMP_NUM_THREADS=2','-e','OPENBLAS_NUM_THREADS=2','-e','MKL_NUM_THREADS=2',
 '-e','PYTHONPATH=/round9/src/sglang-round9-port/python:/round9/runtime-python','--entrypoint','bash','601e76d17e9f','-c',container_script]
subprocess.run(args,check=True)
inspect=json.loads(subprocess.check_output(['docker','inspect',name]))[0];hc=inspect['HostConfig']
assert hc['CpusetCpus']=='480-639' and hc['NanoCpus']==72*10**9
assert hc['Memory']==640*1024**3 and hc['MemorySwap']==hc['Memory']
assert [d['PathOnHost'] for d in hc['Devices'] if d['PathOnHost'].startswith('/dev/davinci')]==['/dev/davinci15','/dev/davinci_manager']
(out/'container-created.json').write_text(json.dumps(inspect,indent=2)+'\n')
with (out/'guard.log').open('w') as log:
 guard=subprocess.Popen(['python3',str(root/'tools/monitor_owned_server.py'),'--container',name,'--output',str(out/'monitor.jsonl'),'--ready-file',str(out/'guard-ready.json'),'--training-pids','4069720','--seconds','3600'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(out/'guard-pid.txt').write_text(str(guard.pid)+'\n')
try:
 deadline=time.monotonic()+30
 while not (out/'guard-ready.json').exists():
  if guard.poll() is not None:raise RuntimeError('guard failed before arming')
  if time.monotonic()>deadline:raise TimeoutError('guard not armed')
  time.sleep(.2)
 subprocess.run(['python3',str(base/'tools/prepare_full_model_trial.py'),'--output',str(out/'preflight')],check=True)
 assert guard.poll() is None
 subprocess.run(['docker','start',name],check=True)
 print('STARTED_MONITORED_TRIAL',name,flush=True)
except BaseException:
 subprocess.run(['docker','stop','--time','10',name],check=False)
 raise
