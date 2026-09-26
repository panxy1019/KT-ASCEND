"""Build immutable Round11 T1 snapshots from the exact v107 HFINAL root."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil


IDENTITIES = {
    ("BOUNDARY", 108),
    ("HYBRID256", 109),
    ("HYBRID512", 110),
    ("HYBRID1024", 111),
    ("HYBRID_CANCEL_REPAIR", 112),
}
GENERATION_LIMITS = {
    "BOUNDARY": 3,
    "HYBRID256": 15,
    "HYBRID512": 4,
    "HYBRID1024": 2,
    "HYBRID_CANCEL_REPAIR": 2,
}
MUTABLE_ROOT_FILES = {
    "execution-budget.json",
    "execution-budget.lock",
    "round11-source-sha256.json",
}


def identity(process, version):
    if (process, version) not in IDENTITIES:
        raise ValueError("Round11 T1 process/version pair mismatch")
    slug = process.lower()
    return {
        "process": process,
        "version": version,
        "container_name": f"kt-round11-t1-{slug}-v{version}",
        "evidence_root": f"/mnt/data1/kt-round9/evidence/20260926-round11-t1-v{version}",
        "remote_snapshot_root": f"/mnt/data1/kt-round9/round11-t1-{slug}-v{version}",
    }


def replace_once(source, old, new, label):
    if source.count(old) != 1:
        raise ValueError(f"{label} precondition mismatch")
    return source.replace(old, new)


def patch_launcher(source, values):
    version = values["version"]
    source = replace_once(
        source,
        "if a.version != 107 or a.cost_mode != 'OFF':",
        f"if a.version != {version} or a.cost_mode != 'OFF':",
        "launcher version",
    )
    source = replace_once(
        source,
        "name='kt-round10-t6-final-v107';out=root/'evidence'/'20260926-round10-t6-v107'",
        f"name='{values['container_name']}';out=root/'evidence'/'20260926-round11-t1-v{version}'",
        "launcher identity",
    )
    return replace_once(
        source,
        "if budget.get('stage')!='ROUND10_T6_FINAL':raise RuntimeError('wrong Round10 T6 budget')",
        f"if budget.get('stage')!='ROUND11_T1_{values['process']}':raise RuntimeError('wrong Round11 T1 budget')",
        "launcher budget stage",
    )


def patch_guard(source):
    source = replace_once(
        source,
        "r'|kt-round10-t6-final-v[0-9]+)', name) is not None",
        "r'|kt-round10-t6-final-v[0-9]+'\n        r'|kt-round11-t1-(?:boundary|hybrid256|hybrid512|hybrid1024|hybrid_cancel_repair)-v[0-9]+)', name) is not None",
        "guard owned name",
    )
    return replace_once(
        source,
        "guard may only control an explicitly allowed owned Round9/Round10 container",
        "guard may only control an explicitly allowed owned Round9/Round10/Round11 container",
        "guard error",
    )


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_manifest(root):
    result = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        relative = str(path.relative_to(root))
        if relative in MUTABLE_ROOT_FILES or relative.startswith("controller-"):
            continue
        result[relative] = sha256(path)
    return result


def controller_source(values, training_id):
    process = values["process"]
    version = values["version"]
    name = values["container_name"]
    out = values["evidence_root"]
    return f'''"""Launch exactly one Round11 T1 {process} model process; no retry."""
import argparse,hashlib,json,pathlib,subprocess,time,traceback
from m3_cleanup import wait_for_guard_exit
p=argparse.ArgumentParser();p.add_argument('--admission',required=True);a=p.parse_args()
root=pathlib.Path(__file__).resolve().parent
base=pathlib.Path('/mnt/data1/kt-round9')
out=pathlib.Path('{out}')
name='{name}'
claim=root/'controller-{process}.json'
manifest=json.loads((root/'round11-source-sha256.json').read_text())
excluded={{'round11-source-sha256.json','execution-budget.json','execution-budget.lock'}}
actual={{str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest() for path in root.rglob('*') if path.is_file() and '__pycache__' not in path.parts and str(path.relative_to(root)) not in excluded and not str(path.relative_to(root)).startswith('controller-')}}
if actual!=manifest:raise RuntimeError('Round11 T1 snapshot manifest mismatch')
with claim.open('x') as stream:json.dump(dict(process='{process}',version={version},started=time.time(),status='LAUNCHING'),stream)
args=['python3',str(root/'source/tools/launch_hybrid.py'),'--version','{version}',
 '--storage','MXFP4','--placement','static','--resource-mode','shared',
 '--shared-training-container','{training_id}',
 '--bank-receipt',str(root/'rebound/bank-receipt.json'),
 '--device-gate',str(base/'evidence/20260920-c128-expert-v19/result.json'),
 '--overlay-python',str(root/'overlay-python'),'--framework-profile',str(root/'framework-profile.json'),
 '--context-target','4096','--context-reference-profile',str(base/'evidence/20260922-plan10-m3-baseline-v64/context-reference-profile.json'),
 '--streaming-profile',str(root/'streaming-profile.json'),'--cancel-trace','--cancel-closure',
 '--cost-mode','OFF','--cost-manifest',str(root/'cost-manifest.json'),
 '--cost-budget',str(root/'execution-budget.json'),'--cost-admission',a.admission]
report=dict(process='{process}',version={version},launch_command=args,status='LAUNCHING',started=time.time())
def save():claim.write_text(json.dumps(report,indent=2)+'\\n')
def cleanup_if_owned():
 try:found=json.loads(subprocess.check_output(['docker','inspect',name],text=True,timeout=30))[0]
 except subprocess.CalledProcessError:return {{'status':'ALREADY_REMOVED'}}
 receipt=out/'container-created.json'
 if not receipt.exists():return {{'status':'NO_VERIFIED_OWNED_CONTAINER_RECEIPT'}}
 expected=json.loads(receipt.read_text())['Id']
 if found['Id']!=expected:raise RuntimeError('owned container identity changed')
 stop_started=time.time();was_running=found['State']['Running']
 if was_running:subprocess.run(['docker','stop','-t','10',expected],check=True,timeout=30)
 ack=wait_for_guard_exit(out/'monitor.jsonl',not_before=stop_started if was_running else report['started'])
 subprocess.run(['docker','rm',expected],check=True,timeout=30)
 return {{'status':'REMOVED_AFTER_GUARD_ACK','guard_exit_ack':ack,'container_id':expected}}
try:
 subprocess.run(args,check=True,timeout=300)
 start=json.loads((out/'model-start.json').read_text())
 report.update(status='SUPERVISING',container_id=start['container_id']);save()
 supervisor=['python3',str(root/'round11_t1_supervisor.py'),'--process','{process}','--version','{version}',
  '--container',start['name'],'--container-id',start['container_id'],'--out',str(out),
  '--driver-root','/round9/'+str(root.relative_to(base))]
 subprocess.run(supervisor,check=True,timeout=21780)
 report['status']=json.loads((out/'round11-t1-supervisor.json').read_text())['status']
except BaseException as error:
 report.update(status='FAIL',error=repr(error),traceback=traceback.format_exc())
 try:report['failure_cleanup']=cleanup_if_owned()
 except BaseException as cleanup:report['cleanup_error']=repr(cleanup)
 raise
finally:
 report['ended']=time.time();save()
'''


def supervisor_source():
    identities = repr(sorted(IDENTITIES))
    limits = repr(GENERATION_LIMITS)
    return f'''"""Bounded Round11 T1 supervisor with guard-ACK-before-remove cleanup."""
import argparse,json,pathlib,subprocess,time,traceback
from m3_cleanup import wait_for_guard_exit
p=argparse.ArgumentParser();p.add_argument('--process',required=True);p.add_argument('--version',type=int,required=True);p.add_argument('--container',required=True);p.add_argument('--container-id',required=True);p.add_argument('--out',type=pathlib.Path,required=True);p.add_argument('--driver-root',required=True);a=p.parse_args()
identities=set(tuple(x) for x in {identities})
limits={limits}
if (a.process,a.version) not in identities:raise ValueError('unregistered Round11 T1 identity')
expected_name=f'kt-round11-t1-{{a.process.lower()}}-v{{a.version}}'
if a.container!=expected_name or len(a.container_id)!=64:raise ValueError('fixed owned container identity required')
def command(args,timeout=30):return subprocess.check_output(args,text=True,timeout=timeout)
def inspect():
 document=json.loads(command(['docker','inspect',a.container]))[0]
 if document['Id']!=a.container_id:raise RuntimeError('owned container identity changed')
 return document
report={{'status':'WAITING_FOR_SERVER','process':a.process,'version':a.version,'container_id':a.container_id,'started_at':time.time(),'generation_driver_starts':0,'readiness_http_requests':0}}
path=a.out/'round11-t1-supervisor.json';path.touch(exist_ok=False)
def save():path.write_text(json.dumps(report,indent=2)+'\\n')
save();child=None
try:
 deadline=time.time()+21600
 while time.time()<deadline:
  document=inspect()
  if not document['State']['Running']:raise RuntimeError('server exited before Round11 T1 completion')
  if any(file.stat().st_size>512*1024**2 for file in a.out.glob('*.log')):raise RuntimeError('log ceiling exceeded')
  log=a.out/'server.log'
  if child is None and log.exists():
   with log.open('rb') as stream:
    stream.seek(max(0,log.stat().st_size-65536));tail=stream.read().decode(errors='replace')
   if 'Uvicorn running on http://127.0.0.1:30091' in tail or 'The server is fired up and ready to roll!' in tail:
    if (a.out/'round11-t1-results.json').exists():raise RuntimeError('Round11 T1 ledger already exists; no retry')
    args=['docker','exec',a.container_id,'python',a.driver_root+'/round11_t1_driver.py','--process',a.process,'--out','/out','--fixture',a.driver_root+'/round11-t1-fixture.json']
    report.update(status='RUNNING_T1',generation_driver_starts=1,driver_command=args);save()
    with (a.out/'round11-t1-driver.log').open('w') as stream:child=subprocess.Popen(args,stdout=stream,stderr=subprocess.STDOUT)
  if child is not None and child.poll() is not None:
   report['driver_exit']=child.returncode
   result=json.loads((a.out/'round11-t1-results.json').read_text())
   generation_limit=limits[a.process];abort_limit=1 if a.process in {{'HYBRID256','HYBRID_CANCEL_REPAIR'}} else 0
   if result.get('process')!=a.process or not 0<=result.get('generation_attempts',-1)<=generation_limit or not 0<=result.get('abort_controls',-1)<=abort_limit or result.get('automatic_retries')!=0:raise RuntimeError('Round11 T1 result budget/identity mismatch')
   if result.get('status')=='PASS_ROUND11_T1_PROCESS' and (result['generation_attempts']!=generation_limit or result['abort_controls']!=abort_limit):raise RuntimeError('passing result did not complete frozen schedule')
   report.update(requests_recorded=len(result['results']),request_status=result['status'])
   report['status']='PASS' if child.returncode==0 and result['status']=='PASS_ROUND11_T1_PROCESS' else 'FAIL';break
  time.sleep(5)
 else:raise TimeoutError('bounded Round11 T1 supervisor deadline')
except BaseException as error:
 report.update(status='FAIL',error=repr(error),traceback=traceback.format_exc())
finally:
 report['ended_at']=time.time();save()
 try:
  document=inspect();stop_started=time.time();was_running=document['State']['Running']
  if was_running:subprocess.run(['docker','stop','-t','10',a.container_id],check=True,timeout=30)
  report['owned_exit_state']=inspect()['State']
  report['guard_exit_ack']=wait_for_guard_exit(a.out/'monitor.jsonl',not_before=stop_started if was_running else report['started_at'])
  subprocess.run(['docker','rm',a.container_id],check=True,timeout=30);report['owned_container_removed']=True
 except BaseException as cleanup:report['cleanup_error']=repr(cleanup);report['status']='FAIL_CLEANUP'
 if child is not None and child.poll() is None:
  try:child.wait(timeout=20)
  except subprocess.TimeoutExpired:child.terminate()
 save()
if report['status']!='PASS':raise SystemExit(1)
'''


def build(args):
    values = identity(args.process, args.version)
    if args.output.exists():
        raise FileExistsError(args.output)
    shutil.copytree(args.v107_root, args.output, ignore=shutil.ignore_patterns("__pycache__"))
    for name in (
        "controller-F.json",
        "execution-budget.lock",
        "round10-source-sha256.json",
        "ROUND10_T6_SNAPSHOT_CONTRACT.json",
    ):
        path = args.output / name
        if path.exists():
            path.unlink()
    launcher = args.output / "source/tools/launch_hybrid.py"
    launcher.write_text(patch_launcher(launcher.read_text(), values))
    guard = args.output / "source/tools/monitor_owned_server.py"
    guard.write_text(patch_guard(guard.read_text()))

    threshold = 129 if args.process == "BOUNDARY" else 1025
    framework_path = args.output / "framework-profile.json"
    framework = json.loads(framework_path.read_text())
    framework["prefill"]["threshold_tokens"] = threshold
    framework_path.write_text(json.dumps(framework, indent=2) + "\n")
    streaming_path = args.output / "streaming-profile.json"
    streaming = json.loads(streaming_path.read_text())
    streaming["threshold_tokens"] = threshold
    streaming_path.write_text(json.dumps(streaming, indent=2) + "\n")

    local_root = pathlib.Path(__file__).resolve().parent
    for name in (
        "round11_t1_protocol.py",
        "round11_t1_driver.py",
        "round11-t1-fixture.json",
    ):
        shutil.copy2(local_root / name, args.output / name)
    (args.output / "host_controller.py").write_text(
        controller_source(values, args.training_id)
    )
    (args.output / "round11_t1_supervisor.py").write_text(supervisor_source())
    budget = {
        "schema": 1,
        "stage": f"ROUND11_T1_{args.process}",
        "process": args.process,
        "version": args.version,
        "model_starts_limit": 1,
        "model_starts_used": 0,
        "starts": [],
        "automatic_retries_limit": 0,
    }
    (args.output / "execution-budget.json").write_text(
        json.dumps(budget, indent=2) + "\n"
    )
    contract = {
        "schema": 1,
        **values,
        "status": "FROZEN_NOT_RUN",
        "parent": "HFINAL_V107",
        "threshold_tokens": threshold,
        "experimental_forced_hybrid": args.process != "BOUNDARY",
        "fixture_sha256": sha256(args.output / "round11-t1-fixture.json"),
        "training_container_id": args.training_id,
        "generation_limit": GENERATION_LIMITS[args.process],
        "abort_controls": 1 if args.process in {"HYBRID256", "HYBRID_CANCEL_REPAIR"} else 0,
        "model_starts_limit": 1,
        "automatic_retries": 0,
        "profiler": "OFF",
        "dynamic": "OFF",
        "request_cost_observer": "HOST_ONLY_PATH_RECEIPT",
    }
    (args.output / "ROUND11_T1_SNAPSHOT_CONTRACT.json").write_text(
        json.dumps(contract, indent=2) + "\n"
    )
    manifest = source_manifest(args.output)
    (args.output / "round11-source-sha256.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(
        "PASS_ROUND11_T1_SNAPSHOT",
        args.process,
        args.version,
        threshold,
        len(manifest),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v107-root", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--process", choices=tuple(GENERATION_LIMITS), required=True)
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--training-id", required=True)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
