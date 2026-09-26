"""Build immutable Round10 T1 process snapshots from the exact v90 root."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil


ALLOWED_IDENTITIES = {("A", 91), ("B", 92), ("A", 93), ("B", 94), ("R", 95)}
MUTABLE_ROOT_FILES = {
    "execution-budget.json",
    "execution-budget.lock",
    "controller-A.json",
    "controller-B.json",
    "round10-source-sha256.json",
}


def identity(process, version):
    if (process, version) not in ALLOWED_IDENTITIES:
        raise ValueError("Round10 T1 process/version pair mismatch")
    return {
        "process": process,
        "version": version,
        "container_name": f"kt-round10-t1-{process.lower()}-v{version}",
        "evidence_root": f"/mnt/data1/kt-round9/evidence/20260926-round10-t1-v{version}",
        "remote_snapshot_root": f"/mnt/data1/kt-round9/round10-t1-h0-v{version}",
    }


def _replace_once(source, old, new, label):
    if source.count(old) != 1:
        raise ValueError(f"launcher {label} precondition mismatch")
    return source.replace(old, new)


def patch_launcher(source, *, process, version):
    values = identity(process, version)
    source = _replace_once(
        source,
        "if a.version != 90 or a.cost_mode != 'OFF':",
        f"if a.version != {version} or a.cost_mode != 'OFF':",
        "version",
    )
    source = _replace_once(
        source,
        "name=f'kt-round9-full-model-v{a.version}';out=root/'evidence'/f'20260925-plan15-threshold-v{a.version}'",
        f"name='{values['container_name']}';out=root/'evidence'/'20260926-round10-t1-v{version}'",
        "identity",
    )
    source = _replace_once(
        source,
        "if budget.get('stage')!='PLAN15_HYBRID_THRESHOLD_CANDIDATE129':raise RuntimeError('wrong independent threshold budget')",
        f"if budget.get('stage')!='ROUND10_T1_PROCESS_{process}':raise RuntimeError('wrong Round10 T1 budget')",
        "budget stage",
    )
    source = _replace_once(
        source,
        "  if a.cost_mode=='ON' and (len(starts)!=1 or starts[0]['mode']!='OFF' or not budget.get('off_verified')):\n   raise RuntimeError('ON requires reviewed OFF result and cleanup')\n",
        "",
        "obsolete ON dependency",
    )
    return source


def patch_guard(source):
    old = """        r'kt-round9-(?:full-model|c128-expert|layerwise|ple|context-tier)-v[0-9]+',
        name) is not None"""
    new = """        r'(?:kt-round9-(?:full-model|c128-expert|layerwise|ple|context-tier)-v[0-9]+'
        r'|kt-round10-t1-[abr]-v[0-9]+)', name) is not None"""
    source = _replace_once(source, old, new, "guard owned-name")
    return _replace_once(
        source,
        "guard may only control an explicitly allowed owned Round9 container",
        "guard may only control an explicitly allowed owned Round9/Round10 container",
        "guard error",
    )


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_manifest(root):
    result = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        rel = str(path.relative_to(root))
        if rel in MUTABLE_ROOT_FILES or rel.startswith("controller-"):
            continue
        result[rel] = sha256(path)
    return result


def controller_source(values, training_id):
    process = values["process"]
    version = values["version"]
    name = values["container_name"]
    out = values["evidence_root"]
    return f'''"""Launch exactly one Round10 T1 {process} model process; no retry."""
import argparse,hashlib,json,pathlib,subprocess,time,traceback
from m3_cleanup import wait_for_guard_exit
p=argparse.ArgumentParser();p.add_argument('--admission',required=True);a=p.parse_args()
root=pathlib.Path(__file__).resolve().parent
base=pathlib.Path('/mnt/data1/kt-round9')
out=pathlib.Path('{out}')
name='{name}'
claim=root/'controller-{process}.json'
manifest=json.loads((root/'round10-source-sha256.json').read_text())
excluded={{'round10-source-sha256.json','execution-budget.json','execution-budget.lock','controller-A.json','controller-B.json'}}
actual={{str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest() for path in root.rglob('*') if path.is_file() and '__pycache__' not in path.parts and str(path.relative_to(root)) not in excluded and not str(path.relative_to(root)).startswith('controller-')}}
if actual!=manifest:raise RuntimeError('Round10 T1 snapshot manifest mismatch')
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
def cleanup_failed_launch():
 receipt=out/'container-created.json'
 if not receipt.exists():return {{'status':'NO_VERIFIED_OWNED_CONTAINER_RECEIPT'}}
 expected=json.loads(receipt.read_text())['Id']
 found=json.loads(subprocess.check_output(['docker','inspect',name],text=True,timeout=30))[0]
 if found['Id']!=expected:raise RuntimeError('failed-launch owned container identity changed')
 stop_started=time.time();was_running=found['State']['Running']
 if was_running:subprocess.run(['docker','stop','-t','10',expected],check=True,timeout=30)
 ack=wait_for_guard_exit(out/'monitor.jsonl',not_before=stop_started if was_running else report['started'])
 subprocess.run(['docker','rm',expected],check=True,timeout=30)
 return {{'status':'REMOVED_AFTER_GUARD_ACK','guard_exit_ack':ack,'container_id':expected}}
try:
 subprocess.run(args,check=True,timeout=300)
 start=json.loads((out/'model-start.json').read_text())
 report.update(status='SUPERVISING',container_id=start['container_id']);save()
 supervisor=['python3',str(root/'round10_t1_supervisor.py'),'--process','{process}','--version','{version}',
  '--container',start['name'],'--container-id',start['container_id'],'--out',str(out),
  '--driver-root','/round9/'+str(root.relative_to(base))]
 subprocess.run(supervisor,check=True,timeout=21780)
 report['status']=json.loads((out/'round10-t1-supervisor.json').read_text())['status']
except BaseException as error:
 report.update(status='FAIL',error=repr(error),traceback=traceback.format_exc())
 try:report['failed_launch_cleanup']=cleanup_failed_launch()
 except BaseException as cleanup:report['cleanup_error']=repr(cleanup)
 raise
finally:
 report['ended']=time.time();save()
'''


def supervisor_source():
    return '''"""Bounded Round10 T1 supervisor with guard-ACK-before-remove cleanup."""
import argparse,json,pathlib,subprocess,time,traceback
from m3_cleanup import wait_for_guard_exit
p=argparse.ArgumentParser();p.add_argument('--process',choices=('A','B','R'),required=True);p.add_argument('--version',type=int,required=True);p.add_argument('--container',required=True);p.add_argument('--container-id',required=True);p.add_argument('--out',type=pathlib.Path,required=True);p.add_argument('--driver-root',required=True);a=p.parse_args()
if (a.process,a.version) not in (('A',91),('B',92),('A',93),('B',94),('R',95)):raise ValueError('unregistered T1 process/version identity')
expected_name=f'kt-round10-t1-{a.process.lower()}-v{a.version}'
if a.container!=expected_name or len(a.container_id)!=64:raise ValueError('fixed owned container identity required')
def command(args,timeout=30):return subprocess.check_output(args,text=True,timeout=timeout)
def inspect():
 document=json.loads(command(['docker','inspect',a.container]))[0]
 if document['Id']!=a.container_id:raise RuntimeError('owned container identity changed')
 return document
report={'status':'WAITING_FOR_SERVER','process':a.process,'version':a.version,'container_id':a.container_id,'started_at':time.time(),'generation_driver_starts':0,'readiness_http_requests':0}
path=a.out/'round10-t1-supervisor.json';path.touch(exist_ok=False)
def save():path.write_text(json.dumps(report,indent=2)+'\\n')
save();child=None
try:
 deadline=time.time()+21600
 while time.time()<deadline:
  document=inspect()
  if not document['State']['Running']:raise RuntimeError('server exited before T1 completion')
  if any(file.stat().st_size>512*1024**2 for file in a.out.glob('*.log')):raise RuntimeError('log ceiling exceeded')
  log=a.out/'server.log'
  if child is None and log.exists():
   with log.open('rb') as stream:
    stream.seek(max(0,log.stat().st_size-65536));tail=stream.read().decode(errors='replace')
   if 'Uvicorn running on http://127.0.0.1:30091' in tail or 'The server is fired up and ready to roll!' in tail:
    if (a.out/'round10-t1-results.json').exists():raise RuntimeError('T1 ledger already exists; no retry')
    args=['docker','exec',a.container_id,'python',a.driver_root+'/round10_t1_driver.py',
     '--process',a.process,'--out','/out','--fixture',a.driver_root+'/round10-t1-fixture.json']
    report.update(status='RUNNING_T1',generation_driver_starts=1,driver_command=args);save()
    with (a.out/'round10-t1-driver.log').open('w') as stream:child=subprocess.Popen(args,stdout=stream,stderr=subprocess.STDOUT)
  if child is not None and child.poll() is not None:
   report['driver_exit']=child.returncode
   result=json.loads((a.out/'round10-t1-results.json').read_text())
   generation_limit={'A':11,'B':11,'R':19}[a.process];abort_limit={'A':1,'B':1,'R':2}[a.process]
   if result.get('process')!=a.process or not 0<=result.get('generation_attempts',-1)<=generation_limit or not 0<=result.get('abort_controls',-1)<=abort_limit or result.get('automatic_retries')!=0:raise RuntimeError('T1 result budget/identity mismatch')
   if result.get('status')=='PASS_ROUND10_T1_PROCESS' and (result['generation_attempts']!=generation_limit or result['abort_controls']!=abort_limit):raise RuntimeError('passing T1 result did not complete its frozen schedule')
   report.update(requests_recorded=len(result['results']),request_status=result['status'])
   report['status']='PASS' if child.returncode==0 and result['status']=='PASS_ROUND10_T1_PROCESS' else 'FAIL';break
  time.sleep(5)
 else:raise TimeoutError('bounded T1 supervisor deadline')
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
    shutil.copytree(args.v90_root, args.output, ignore=shutil.ignore_patterns("__pycache__"))
    for name in ("controller-CANDIDATE129.json", "execution-budget.lock"):
        path = args.output / name
        if path.exists():
            path.unlink()
    launcher = args.output / "source/tools/launch_hybrid.py"
    launcher.write_text(
        patch_launcher(launcher.read_text(), process=args.process, version=args.version)
    )
    guard = args.output / "source/tools/monitor_owned_server.py"
    guard.write_text(patch_guard(guard.read_text()))
    if args.training_id not in launcher.read_text():
        old = "0419bcfee46f317d6833e038c28e81cf3d3ecf47323a423062bac2764e0ab2ef"
        text = launcher.read_text()
        if text.count(old) != 1:
            raise ValueError("training identity precondition mismatch")
        launcher.write_text(text.replace(old, args.training_id))
    local_root = pathlib.Path(__file__).resolve().parent
    for name in ("round10_t1_protocol.py", "round10_t1_driver.py", "round10-t1-fixture.json"):
        shutil.copy2(local_root / name, args.output / name)
    (args.output / "host_controller.py").write_text(
        controller_source(values, args.training_id)
    )
    (args.output / "round10_t1_supervisor.py").write_text(supervisor_source())
    budget = {
        "schema": 1,
        "stage": f"ROUND10_T1_PROCESS_{args.process}",
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
        "parent": "R9_V90_RAW",
        "parent_source_manifest": "source-sha256.json",
        "fixture_sha256": sha256(args.output / "round10-t1-fixture.json"),
        "training_container_id": args.training_id,
        "generation_limit": 19 if args.process == "R" else 11,
        "abort_controls": 2 if args.process == "R" else 1,
        "automatic_retries": 0,
        "profiler": "OFF",
        "dynamic": "OFF",
        "request_cost_observer": "OFF",
    }
    (args.output / "ROUND10_T1_SNAPSHOT_CONTRACT.json").write_text(
        json.dumps(contract, indent=2) + "\n"
    )
    manifest = source_manifest(args.output)
    (args.output / "round10-source-sha256.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(
        "PASS_ROUND10_T1_SNAPSHOT",
        args.process,
        args.version,
        len(manifest),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v90-root", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--process", choices=("A", "B", "R"), required=True)
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--training-id", required=True)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
