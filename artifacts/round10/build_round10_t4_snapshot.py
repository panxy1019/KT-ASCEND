"""Build one immutable Round10 D1 Dynamic snapshot from qualified H0 v95."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
import sys


MUTABLE = {"round10-source-sha256.json", "execution-budget.json", "execution-budget.lock", "host-controller.json"}
RUNTIME_FILES = ("prefill_binding.py", "prefill_end_boundary.py", "dynamic_boundary.py", "cancel_lifecycle.py")


def _replace_once(source, old, new, label):
    if source.count(old) != 1:
        raise ValueError(f"{label} precondition mismatch")
    return source.replace(old, new)


def patch_launcher(source, version):
    source = _replace_once(source, "if a.version != 95 or a.cost_mode != 'OFF':",
                           f"if a.version != {version} or a.cost_mode != 'OFF':", "launcher version")
    source = _replace_once(source, "name='kt-round10-t1-r-v95';out=root/'evidence'/'20260926-round10-t1-v95'",
                           f"name='kt-round10-t4-d-v{version}';out=root/'evidence'/'20260926-round10-t4-v{version}'",
                           "launcher identity")
    return _replace_once(source, "if budget.get('stage')!='ROUND10_T1_PROCESS_R':raise RuntimeError('wrong Round10 T1 budget')",
                         "if budget.get('stage')!='ROUND10_T4_D1':raise RuntimeError('wrong Round10 T4 budget')",
                         "launcher budget")


def patch_guard(source):
    old = "r'|kt-round10-t1-[abr]-v[0-9]+)', name) is not None"
    new = "r'|kt-round10-t1-[abr]-v[0-9]+'\n        r'|kt-round10-t4-d-v[0-9]+)', name) is not None"
    return _replace_once(source, old, new, "guard identity")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_manifest(root):
    return {
        str(path.relative_to(root)): sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
        and str(path.relative_to(root)) not in MUTABLE
        and not str(path.relative_to(root)).startswith("host-controller")
    }


def refresh_overlay(root):
    tools = root / "source/tools"
    sys.path.insert(0, str(tools))
    try:
        from install_candidate_overlay import tree_digest, tree_files
        overlay = root / "overlay-python"
        receipt_path = overlay / "round9-candidate-overlay.json"
        receipt = json.loads(receipt_path.read_text())
        for name in RUNTIME_FILES:
            source = root / "source/runtime" / name
            target = overlay / "sglang/srt/layers/round9" / name
            shutil.copy2(source, target)
            receipt.setdefault("modules", {})[name] = sha256(source)
        receipt["candidate_files"] = tree_files(overlay)
        receipt["candidate_tree_sha256"] = tree_digest(receipt["candidate_files"])
        receipt["round10_dynamic_wiring"] = {
            "status": "INSTALLED_NOT_FULL_MODEL_QUALIFIED",
            "runtime_files": {name: sha256(root / "source/runtime" / name) for name in RUNTIME_FILES},
        }
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    finally:
        sys.path.remove(str(tools))


def controller_source(version, training_id):
    return f'''"""Launch exactly one Round10 T4 D1 model process; no retry."""
import argparse,hashlib,json,pathlib,subprocess,time,traceback
from m3_cleanup import wait_for_guard_exit
p=argparse.ArgumentParser();p.add_argument('--admission',required=True);a=p.parse_args()
root=pathlib.Path(__file__).resolve().parent;base=pathlib.Path('/mnt/data1/kt-round9')
out=base/'evidence'/'20260926-round10-t4-v{version}';name='kt-round10-t4-d-v{version}'
claim=root/'host-controller.json';manifest=json.loads((root/'round10-source-sha256.json').read_text())
excluded={{'round10-source-sha256.json','execution-budget.json','execution-budget.lock','host-controller.json'}}
actual={{str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest() for path in root.rglob('*') if path.is_file() and '__pycache__' not in path.parts and str(path.relative_to(root)) not in excluded and not str(path.relative_to(root)).startswith('host-controller')}}
if actual!=manifest:raise RuntimeError('Round10 T4 snapshot manifest mismatch')
with claim.open('x') as stream:json.dump(dict(version={version},started=time.time(),status='LAUNCHING'),stream)
args=['python3',str(root/'source/tools/launch_hybrid.py'),'--version','{version}',
 '--storage','MXFP4','--placement','request_boundary','--resource-mode','shared',
 '--shared-training-container','{training_id}', '--bank-receipt',str(root/'rebound/bank-receipt.json'),
 '--device-gate',str(base/'evidence/20260920-c128-expert-v19/result.json'),
 '--overlay-python',str(root/'overlay-python'),'--framework-profile',str(root/'framework-profile.json'),
 '--context-target','4096','--context-reference-profile',str(base/'evidence/20260922-plan10-m3-baseline-v64/context-reference-profile.json'),
 '--streaming-profile',str(root/'streaming-profile.json'),'--cancel-trace','--cancel-closure',
 '--cost-mode','OFF','--cost-manifest',str(root/'cost-manifest.json'),
 '--cost-budget',str(root/'execution-budget.json'),'--cost-admission',a.admission]
report=dict(version={version},launch_command=args,status='LAUNCHING',started=time.time())
def save():claim.write_text(json.dumps(report,indent=2)+'\\n')
def cleanup_failed_launch():
 receipt=out/'container-created.json'
 if not receipt.exists():return {{'status':'NO_VERIFIED_OWNED_CONTAINER_RECEIPT'}}
 expected=json.loads(receipt.read_text())['Id'];found=json.loads(subprocess.check_output(['docker','inspect',name],text=True,timeout=30))[0]
 if found['Id']!=expected:raise RuntimeError('failed-launch owned container identity changed')
 stop_started=time.time();was_running=found['State']['Running']
 if was_running:subprocess.run(['docker','stop','-t','10',expected],check=True,timeout=30)
 ack=wait_for_guard_exit(out/'monitor.jsonl',not_before=stop_started if was_running else report['started'])
 subprocess.run(['docker','rm',expected],check=True,timeout=30)
 return {{'status':'REMOVED_AFTER_GUARD_ACK','guard_exit_ack':ack,'container_id':expected}}
try:
 subprocess.run(args,check=True,timeout=300)
 start=json.loads((out/'model-start.json').read_text());report.update(status='SUPERVISING',container_id=start['container_id']);save()
 supervisor=['python3',str(root/'round10_t4_supervisor.py'),'--version','{version}','--container',start['name'],
  '--container-id',start['container_id'],'--out',str(out),'--driver-root','/round9/'+str(root.relative_to(base))]
 subprocess.run(supervisor,check=True,timeout=21780)
 report['status']=json.loads((out/'round10-t4-supervisor.json').read_text())['status']
except BaseException as error:
 report.update(status='FAIL',error=repr(error),traceback=traceback.format_exc())
 try:report['failed_launch_cleanup']=cleanup_failed_launch()
 except BaseException as cleanup:report['cleanup_error']=repr(cleanup)
 raise
finally:report['ended']=time.time();save()
'''


def supervisor_source(version):
    return f'''"""Bounded T4 D1 supervisor with guard-ACK-before-remove cleanup."""
import argparse,json,pathlib,subprocess,time,traceback
from m3_cleanup import wait_for_guard_exit
p=argparse.ArgumentParser();p.add_argument('--version',type=int,required=True);p.add_argument('--container',required=True);p.add_argument('--container-id',required=True);p.add_argument('--out',type=pathlib.Path,required=True);p.add_argument('--driver-root',required=True);a=p.parse_args()
if a.version!={version} or a.container!='kt-round10-t4-d-v{version}' or len(a.container_id)!=64:raise ValueError('fixed T4 identity required')
def command(args,timeout=30):return subprocess.check_output(args,text=True,timeout=timeout)
def inspect():
 document=json.loads(command(['docker','inspect',a.container]))[0]
 if document['Id']!=a.container_id:raise RuntimeError('owned container identity changed')
 return document
report={{'status':'WAITING_FOR_SERVER','version':a.version,'container_id':a.container_id,'started_at':time.time(),'generation_driver_starts':0}}
path=a.out/'round10-t4-supervisor.json';path.touch(exist_ok=False)
def save():path.write_text(json.dumps(report,indent=2)+'\\n')
save();child=None
try:
 deadline=time.time()+21600
 while time.time()<deadline:
  document=inspect()
  if not document['State']['Running']:raise RuntimeError('server exited before T4 completion')
  if any(file.stat().st_size>512*1024**2 for file in a.out.glob('*.log')):raise RuntimeError('log ceiling exceeded')
  log=a.out/'server.log'
  if child is None and log.exists():
   with log.open('rb') as stream:stream.seek(max(0,log.stat().st_size-65536));tail=stream.read().decode(errors='replace')
   if 'Uvicorn running on http://127.0.0.1:30091' in tail or 'The server is fired up and ready to roll!' in tail:
    if (a.out/'round10-t4-results.json').exists():raise RuntimeError('T4 ledger already exists; no retry')
    args=['docker','exec',a.container_id,'python',a.driver_root+'/round10_t4_driver.py','--out','/out','--fixture',a.driver_root+'/round10-t1-fixture.json']
    report.update(status='RUNNING_T4',generation_driver_starts=1,driver_command=args);save()
    with (a.out/'round10-t4-driver.log').open('w') as stream:child=subprocess.Popen(args,stdout=stream,stderr=subprocess.STDOUT)
  if child is not None and child.poll() is not None:
   report['driver_exit']=child.returncode;result=json.loads((a.out/'round10-t4-results.json').read_text())
   if result.get('generation_attempts')!=5 or result.get('abort_controls')!=1 or result.get('automatic_retries')!=0:raise RuntimeError('T4 result budget mismatch')
   report['status']='PASS' if child.returncode==0 and result.get('status')=='PASS_ROUND10_T4_D1_PROCESS' else 'FAIL';break
  time.sleep(5)
 else:raise TimeoutError('bounded T4 supervisor deadline')
except BaseException as error:report.update(status='FAIL',error=repr(error),traceback=traceback.format_exc())
finally:
 report['ended_at']=time.time();save()
 try:
  document=inspect();stop_started=time.time();was_running=document['State']['Running']
  if was_running:subprocess.run(['docker','stop','-t','10',a.container_id],check=True,timeout=30)
  report['owned_exit_state']=inspect()['State'];report['guard_exit_ack']=wait_for_guard_exit(a.out/'monitor.jsonl',not_before=stop_started if was_running else report['started_at'])
  subprocess.run(['docker','rm',a.container_id],check=True,timeout=30);report['owned_container_removed']=True
 except BaseException as cleanup:report['cleanup_error']=repr(cleanup);report['status']='FAIL_CLEANUP'
 if child is not None and child.poll() is None:
  try:child.wait(timeout=20)
  except subprocess.TimeoutExpired:child.terminate()
 save()
if report['status']!='PASS':raise SystemExit(1)
'''


def build(args):
    if args.output.exists(): raise FileExistsError(args.output)
    shutil.copytree(args.parent, args.output, ignore=shutil.ignore_patterns("__pycache__"))
    for name in ("controller-R.json", "execution-budget.lock", "host-controller.json"):
        path = args.output / name
        if path.exists(): path.unlink()
    launcher = args.output / "source/tools/launch_hybrid.py"
    launcher.write_text(patch_launcher(launcher.read_text(), args.version))
    guard = args.output / "source/tools/monitor_owned_server.py"
    guard.write_text(patch_guard(guard.read_text()))
    local = pathlib.Path(__file__).resolve().parents[2]
    for name in RUNTIME_FILES:
        shutil.copy2(local / "migration_bridge/round9/runtime" / name, args.output / "source/runtime" / name)
    for name in ("test_prefill_binding.py", "test_prefill_end_boundary.py", "test_dynamic_boundary.py", "test_cancel_lifecycle.py"):
        shutil.copy2(local / "migration_bridge/round9/tests" / name, args.output / "source/tests" / name)
    artifact = pathlib.Path(__file__).resolve().parent
    for name in ("round10_t1_driver.py", "round10_t1_protocol.py", "round10_t4_driver.py", "round10-t1-fixture.json"):
        shutil.copy2(artifact / name, args.output / name)
    shutil.copy2(args.profile, args.output / "rebound/candidate-profile.json")
    shutil.copy2(args.bank_receipt, args.output / "rebound/bank-receipt.json")
    framework = json.loads((args.output / "framework-profile.json").read_text())
    framework["prefill"]["update_residents"] = True
    (args.output / "framework-profile.json").write_text(json.dumps(framework, indent=2) + "\n")
    refresh_overlay(args.output)
    (args.output / "host_controller.py").write_text(controller_source(args.version, args.training_id))
    (args.output / "round10_t4_supervisor.py").write_text(supervisor_source(args.version))
    budget = {"schema":1,"stage":"ROUND10_T4_D1","version":args.version,"model_starts_limit":1,"model_starts_used":0,"starts":[],"automatic_retries_limit":0}
    (args.output / "execution-budget.json").write_text(json.dumps(budget, indent=2) + "\n")
    contract = {"schema":1,"status":"FROZEN_NOT_RUN","version":args.version,"parent":"H0_ROUND10_V95",
                "only_runtime_change":"HYBRID_PREFILL_STATISTICS_TO_EXISTING_DYNAMIC_TRANSACTION",
                "dynamic":"PREFILL_END_ON","generation_limit":5,"abort_limit":1,"automatic_retries":0,
                "profile_sha256":sha256(args.profile),"bank_receipt_sha256":sha256(args.bank_receipt)}
    (args.output / "ROUND10_T4_SNAPSHOT_CONTRACT.json").write_text(json.dumps(contract, indent=2) + "\n")
    manifest = source_manifest(args.output)
    (args.output / "round10-source-sha256.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print("PASS_ROUND10_T4_SNAPSHOT", args.version, len(manifest), flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',type=pathlib.Path,required=True);parser.add_argument('--output',type=pathlib.Path,required=True)
    parser.add_argument('--profile',type=pathlib.Path,required=True);parser.add_argument('--bank-receipt',type=pathlib.Path,required=True)
    parser.add_argument('--version',type=int,default=102);parser.add_argument('--training-id',required=True)
    build(parser.parse_args())


if __name__=='__main__':main()
