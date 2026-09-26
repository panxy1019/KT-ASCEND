#!/usr/bin/env python3
"""Build one immutable Round11 T6 HFINAL reproduction snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import stat


TRAINING_ID = "0419bcfee46f317d6833e038c28e81cf3d3ecf47323a423062bac2764e0ab2ef"
ROLE_CONTRACT = {
    147: ("BASELINE", "A"),
    148: ("FINAL", "A"),
    149: ("FINAL", "B"),
    150: ("BASELINE", "B"),
}
MUTABLE = {
    "round11-source-sha256.json",
    "execution-budget.json",
    "execution-budget.lock",
    "host-controller.json",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def role(version: int) -> tuple[str, str]:
    if version not in ROLE_CONTRACT:
        raise ValueError("T6 version identity mismatch")
    return ROLE_CONTRACT[version]


def replace_once(source: str, old: str, new: str, label: str) -> str:
    if source.count(old) != 1:
        raise ValueError(f"{label} precondition mismatch")
    return source.replace(old, new)


def patch_launcher(source: str, *, version: int) -> str:
    mode, unused_block = role(version)
    name = f"kt-round11-t6-{mode.lower()}-v{version}"
    source = replace_once(
        source,
        "if a.version != 127 or a.cost_mode != 'OFF':",
        f"if a.version != {version} or a.cost_mode != 'OFF':",
        "T6 launcher version",
    )
    source = replace_once(
        source,
        "name='kt-round11-t4-static-v127';out=root/'evidence'/'20260926-round11-t4-v127'",
        f"name='{name}';out=root/'evidence'/'20260926-round11-t6-v{version}'",
        "T6 launcher identity",
    )
    source = replace_once(
        source,
        "if budget.get('stage')!='ROUND11_T4_PROCESS':raise RuntimeError('wrong Round11 T4 budget')",
        "if budget.get('stage')!='ROUND11_T6_PROCESS':raise RuntimeError('wrong Round11 T6 budget')",
        "T6 launcher budget",
    )
    compile(source, "launch_hybrid.py", "exec")
    return source


def source_manifest(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and "__pycache__" not in path.parts
        and str(path.relative_to(root)) not in MUTABLE
        and not str(path.relative_to(root)).startswith("host-controller")
    }


def controller_source(*, version: int) -> str:
    mode, block = role(version)
    template = '''"""Launch one immutable Round11 T6 __MODE__-__BLOCK__ process."""
import argparse,hashlib,json,pathlib,subprocess,time,traceback
from m3_cleanup import wait_for_guard_exit
p=argparse.ArgumentParser();p.add_argument('--admission',required=True);a=p.parse_args()
root=pathlib.Path(__file__).resolve().parent;base=pathlib.Path('/mnt/data1/kt-round9')
out=base/'evidence'/'20260926-round11-t6-v__VERSION__';name='kt-round11-t6-__MODE_LOWER__-v__VERSION__'
claim=root/'host-controller.json';manifest=json.loads((root/'round11-source-sha256.json').read_text())
excluded={'round11-source-sha256.json','execution-budget.json','execution-budget.lock','host-controller.json'}
actual={str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest() for path in root.rglob('*') if path.is_file() and '__pycache__' not in path.parts and str(path.relative_to(root)) not in excluded and not str(path.relative_to(root)).startswith('host-controller')}
if actual!=manifest:raise RuntimeError('Round11 T6 snapshot manifest mismatch')
with claim.open('x') as stream:json.dump(dict(mode='__MODE__',block='__BLOCK__',version=__VERSION__,started=time.time(),status='LAUNCHING'),stream)
args=['python3',str(root/'source/tools/launch_hybrid.py'),'--version','__VERSION__',
 '--storage','MXFP4','--placement','static','--resource-mode','shared',
 '--shared-training-container','__TRAINING_ID__','--bank-receipt',str(root/'rebound/bank-receipt.json'),
 '--device-gate',str(base/'evidence/20260920-c128-expert-v19/result.json'),
 '--overlay-python',str(root/'overlay-python'),'--framework-profile',str(root/'framework-profile.json'),
 '--context-target','4096','--context-reference-profile',str(base/'evidence/20260922-plan10-m3-baseline-v64/context-reference-profile.json'),
 '--streaming-profile',str(root/'streaming-profile.json'),'--cancel-trace','--cancel-closure',
 '--cost-mode','OFF','--cost-manifest',str(root/'cost-manifest.json'),
 '--cost-budget',str(root/'execution-budget.json'),'--cost-admission',a.admission]
report=dict(mode='__MODE__',block='__BLOCK__',version=__VERSION__,launch_command=args,status='LAUNCHING',started=time.time())
def save():claim.write_text(json.dumps(report,indent=2)+'\\n')
def cleanup_failed_launch():
 receipt=out/'container-created.json'
 if not receipt.exists():return {'status':'NO_VERIFIED_OWNED_CONTAINER_RECEIPT'}
 expected=json.loads(receipt.read_text())['Id'];found=json.loads(subprocess.check_output(['docker','inspect',name],text=True,timeout=30))[0]
 if found['Id']!=expected:raise RuntimeError('failed-launch owned container identity changed')
 stop_started=time.time();was_running=found['State']['Running']
 if was_running:subprocess.run(['docker','stop','-t','10',expected],check=True,timeout=30)
 ack=wait_for_guard_exit(out/'monitor.jsonl',not_before=stop_started if was_running else report['started'])
 subprocess.run(['docker','rm',expected],check=True,timeout=30)
 return {'status':'REMOVED_AFTER_GUARD_ACK','guard_exit_ack':ack,'container_id':expected}
try:
 subprocess.run(args,check=True,timeout=300)
 start=json.loads((out/'model-start.json').read_text());report.update(status='SUPERVISING',container_id=start['container_id']);save()
 supervisor=['python3',str(root/'round11_t6_supervisor.py'),'--version','__VERSION__','--mode','__MODE__','--block','__BLOCK__',
  '--container',start['name'],'--container-id',start['container_id'],'--out',str(out),
  '--driver-root','/round9/'+str(root.relative_to(base))]
 subprocess.run(supervisor,check=True,timeout=21780)
 report['status']=json.loads((out/'round11-t6-supervisor.json').read_text())['status']
except BaseException as error:
 report.update(status='FAIL',error=repr(error),traceback=traceback.format_exc())
 try:report['failed_launch_cleanup']=cleanup_failed_launch()
 except BaseException as cleanup:report['cleanup_error']=repr(cleanup)
 raise
finally:report['ended']=time.time();save()
'''
    replacements = {
        "__VERSION__": str(version),
        "__MODE__": mode,
        "__MODE_LOWER__": mode.lower(),
        "__BLOCK__": block,
        "__TRAINING_ID__": TRAINING_ID,
    }
    for old, new in replacements.items():
        template = template.replace(old, new)
    compile(template, "host_controller.py", "exec")
    return template


def supervisor_source(*, version: int) -> str:
    mode, block = role(version)
    template = '''"""Supervise one immutable Round11 T6 __MODE__-__BLOCK__ process."""
import argparse,json,pathlib,subprocess,time,traceback
from m3_cleanup import wait_for_guard_exit
p=argparse.ArgumentParser();p.add_argument('--version',type=int,required=True);p.add_argument('--mode',required=True);p.add_argument('--block',required=True);p.add_argument('--container',required=True);p.add_argument('--container-id',required=True);p.add_argument('--out',type=pathlib.Path,required=True);p.add_argument('--driver-root',required=True);a=p.parse_args()
if (a.version,a.mode,a.block,a.container)!=(__VERSION__,'__MODE__','__BLOCK__','kt-round11-t6-__MODE_LOWER__-v__VERSION__') or len(a.container_id)!=64:raise ValueError('fixed Round11 T6 identity required')
def command(args,timeout=30):return subprocess.check_output(args,text=True,timeout=timeout)
def inspect():
 document=json.loads(command(['docker','inspect',a.container]))[0]
 if document['Id']!=a.container_id:raise RuntimeError('owned container identity changed')
 return document
report={'status':'WAITING_FOR_SERVER','version':a.version,'mode':a.mode,'block':a.block,'container_id':a.container_id,'started_at':time.time(),'generation_driver_starts':0}
path=a.out/'round11-t6-supervisor.json';path.touch(exist_ok=False)
def save():path.write_text(json.dumps(report,indent=2)+'\\n')
save();child=None;raw=None
try:
 deadline=time.time()+21600
 while time.time()<deadline:
  document=inspect()
  if not document['State']['Running']:raise RuntimeError('server exited before T6 completion')
  if any(file.stat().st_size>512*1024**2 for file in a.out.glob('*.log')):raise RuntimeError('log ceiling exceeded')
  log=a.out/'server.log'
  if child is None and log.exists():
   with log.open('rb') as stream:stream.seek(max(0,log.stat().st_size-65536));tail=stream.read().decode(errors='replace')
   if 'Uvicorn running on http://127.0.0.1:30091' in tail or 'The server is fired up and ready to roll!' in tail:
    if (a.out/'round11-t6-raw.json').exists():raise RuntimeError('T6 ledger already exists; no retry')
    args=['docker','exec',a.container_id,'python',a.driver_root+'/round11_t6_driver.py','--mode',a.mode,'--block',a.block,'--version',str(a.version),'--out','/out','--fixture',a.driver_root+'/round10-t1-fixture.json']
    report.update(status='RUNNING_T6',generation_driver_starts=1,driver_command=args);save()
    with (a.out/'round11-t6-driver.log').open('x') as stream:child=subprocess.Popen(args,stdout=stream,stderr=subprocess.STDOUT)
  if child is not None and child.poll() is not None:
   report['driver_exit']=child.returncode;raw_path=a.out/'round11-t6-raw.json'
   if not raw_path.exists():raise RuntimeError('T6 driver exited before publishing raw result')
   raw=json.loads(raw_path.read_text())
   expected_abort=int((a.mode,a.block) in {('BASELINE','A'),('FINAL','B')})
   if raw.get('generation_attempts')!=6 or raw.get('abort_controls')!=expected_abort or raw.get('automatic_retries')!=0 or raw.get('profiler')!='OFF':raise RuntimeError('T6 result budget mismatch')
   report['status']='DRIVER_PASS_WAITING_CLEANUP' if child.returncode==0 and raw.get('status')=='PASS_ROUND11_T6_DRIVER_RAW' else 'FAIL'
   break
  time.sleep(5)
 else:raise TimeoutError('bounded T6 supervisor deadline')
except BaseException as error:report.update(status='FAIL',error=repr(error),traceback=traceback.format_exc())
finally:
 report['ended_at']=time.time();save()
 try:
  document=inspect();stop_started=time.time();was_running=document['State']['Running']
  if was_running:subprocess.run(['docker','stop','-t','10',a.container_id],check=True,timeout=30)
  report['owned_exit_state']=inspect()['State'];report['guard_exit_ack']=wait_for_guard_exit(a.out/'monitor.jsonl',not_before=stop_started if was_running else report['started_at'])
  subprocess.run(['docker','rm',a.container_id],check=True,timeout=30);report['owned_container_removed']=True
  proc=command(['npu-smi','info','-t','proc-mem','-i','6','-c','0']);(a.out/'post-cleanup-proc-mem.txt').write_text(proc)
  if 'No process in device' not in proc:raise RuntimeError('physical12 is not empty after T6 cleanup')
  report['physical12_after']='EMPTY'
 except BaseException as cleanup:report['cleanup_error']=repr(cleanup);report['status']='FAIL_CLEANUP'
 if child is not None and child.poll() is None:
  try:child.wait(timeout=20)
  except subprocess.TimeoutExpired:child.terminate()
 if report['status']=='DRIVER_PASS_WAITING_CLEANUP' and raw is not None:
  final=dict(raw);final['status']='PASS_ROUND11_T6_PROCESS';final['cleanup']={'guard_exit_ack':True,'container_removed':True,'physical12_after':'EMPTY'}
  (a.out/'round11-t6-final.json').write_text(json.dumps(final,indent=2)+'\\n');report['status']='PASS_ROUND11_T6_PROCESS'
 save()
if report['status']!='PASS_ROUND11_T6_PROCESS':raise SystemExit(1)
'''
    replacements = {
        "__VERSION__": str(version),
        "__MODE__": mode,
        "__MODE_LOWER__": mode.lower(),
        "__BLOCK__": block,
    }
    for old, new in replacements.items():
        template = template.replace(old, new)
    compile(template, "round11_t6_supervisor.py", "exec")
    return template


def build(parent: Path, output: Path, *, version: int) -> Path:
    mode, block = role(version)
    if output.exists():
        raise FileExistsError(output)
    local = Path(__file__).resolve().parents[2]
    shutil.copytree(parent, output, ignore=shutil.ignore_patterns("__pycache__"))
    for path in (output, *output.rglob("*")):
        path.chmod(path.stat().st_mode | stat.S_IWUSR)
    for name in ("controller-F.json", "execution-budget.lock", "host-controller.json"):
        path = output / name
        if path.exists():
            path.unlink()
    launcher = output / "source/tools/launch_hybrid.py"
    launcher.write_text(patch_launcher(launcher.read_text(), version=version))
    shutil.copy2(local / "artifacts/round11/round11_monitor_owned.py", output / "source/tools/monitor_owned_server.py")
    shutil.copy2(local / "artifacts/round11/round11_t4_driver.py", output / "round11_t4_driver.py")
    shutil.copy2(local / "artifacts/round11/round11_t6_driver.py", output / "round11_t6_driver.py")
    (output / "host_controller.py").write_text(controller_source(version=version))
    (output / "round11_t6_supervisor.py").write_text(supervisor_source(version=version))
    original_qsa = output / "overlay-python/sglang/srt/layers/round9/qsa.py"
    if sha256(original_qsa) != "76e9f34066f0b42f42545427967c0b789d5c32031dc5c085e230efaf95e3c54c":
        raise ValueError("T6 HFINAL QSA provider drifted")
    budget = {
        "schema": 1,
        "stage": "ROUND11_T6_PROCESS",
        "version": version,
        "mode": mode,
        "block": block,
        "model_starts_limit": 1,
        "model_starts_used": 0,
        "starts": [],
        "generation_limit": 6,
        "abort_limit": int((mode, block) in {("BASELINE", "A"), ("FINAL", "B")}),
        "automatic_retries_limit": 0,
    }
    (output / "execution-budget.json").write_text(json.dumps(budget, indent=2) + "\n")
    contract = {
        "schema": 1,
        "status": "FROZEN_NOT_RUN",
        "stage": "ROUND11_T6_FINAL_HFINAL_REPRODUCTION",
        "version": version,
        "mode": mode,
        "block": block,
        "runtime_identity": "ROUND11_HFINAL_EQUALS_V107_ORIGINAL_PROVIDERS",
        "qsa_candidate": "COMPONENT_ONLY_NO_PROMOTION",
        "profiler": "OFF",
        "generation_limit": 6,
        "abort_limit": budget["abort_limit"],
        "automatic_retries": 0,
        "performance_requests": [
            {"input_tokens": 128, "decode_tokens": value, "ignore_eos": True}
            for value in (32, 64, 128)
        ],
        "unchanged": [
            "model", "tokenizer", "MXFP4 bank", "Static Q32", "threshold129",
            "chunk32", "PLE ASYNC", "Dynamic OFF", "C128 NUMA2/3/4/7",
            "original GDN", "original QSA", "original PLE",
        ],
    }
    (output / "ROUND11_T6_SNAPSHOT_CONTRACT.json").write_text(json.dumps(contract, indent=2) + "\n")
    manifest = source_manifest(output)
    (output / "round11-source-sha256.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version", type=int, required=True)
    args = parser.parse_args()
    root = build(args.parent, args.output, version=args.version)
    print("PASS_ROUND11_T6_SNAPSHOT", args.version, role(args.version), sha256(root / "round11-source-sha256.json"), flush=True)


if __name__ == "__main__":
    main()
