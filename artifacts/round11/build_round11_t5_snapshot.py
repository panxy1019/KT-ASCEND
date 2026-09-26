#!/usr/bin/env python3
"""Build one immutable Round11 GDN/QSA/PLE attribution snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import stat
import sys


TRAINING_ID = "0419bcfee46f317d6833e038c28e81cf3d3ecf47323a423062bac2764e0ab2ef"
PARENT_IDENTITY = "ROUND11_HFINAL_V107_ORIGINAL_C128_MXFP4"
ROLE_CONTRACT = {"A": (136, 138, 139), "B": (137, 140)}
MUTABLE = {
    "round11-source-sha256.json",
    "execution-budget.json",
    "execution-budget.lock",
    "host-controller.json",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(source: str, old: str, new: str, label: str) -> str:
    if source.count(old) != 1:
        raise ValueError(f"{label} precondition mismatch")
    return source.replace(old, new)


def patch_scheduler(source: str) -> str:
    old = (
        "                from sglang.srt.layers.round9.framework_hooks import framework_before\n"
        "                framework_before(self, batch)\n"
        "                result = self.run_batch(batch)\n"
        "                self.process_batch_result(batch, result)\n"
    )
    new = (
        "                from sglang.srt.layers.round9.framework_hooks import framework_before\n"
        "                framework_before(self, batch)\n"
        "                from sglang.srt.layers.round9.round11_b_attribution import (\n"
        "                    before as round11_b_before, after as round11_b_after)\n"
        "                round11_b_before(self, batch)\n"
        "                result = self.run_batch(batch)\n"
        "                round11_b_after(self)\n"
        "                self.process_batch_result(batch, result)\n"
    )
    return replace_once(source, old, new, "Round11 B scheduler hook")


def patch_launcher(source: str, *, block: str, version: int) -> str:
    name = f"kt-round11-t5-hotspot-v{version}"
    source = replace_once(
        source,
        "if a.version != 127 or a.cost_mode != 'OFF':",
        f"if a.version != {version} or a.cost_mode != 'OFF':",
        "T5 launcher version",
    )
    source = replace_once(
        source,
        "name='kt-round11-t4-static-v127';out=root/'evidence'/'20260926-round11-t4-v127'",
        f"name='{name}';out=root/'evidence'/'20260926-round11-t5-v{version}'",
        "T5 launcher identity",
    )
    old_env = (
        "'ROUND9_DYNAMIC_EVENTS=/out/dynamic-events.jsonl',"
        "f'ROUND9_DYNAMIC_BOUNDARY_ENABLE={int(a.placement==\"request_boundary\")}',"
        "f'ROUND11_DYNAMIC_TIMING_ENABLE={int(a.placement==\"request_boundary\")}'):args+=['-e',env]"
    )
    new_env = (
        "'ROUND9_DYNAMIC_EVENTS=/out/dynamic-events.jsonl',"
        "f'ROUND9_DYNAMIC_BOUNDARY_ENABLE={int(a.placement==\"request_boundary\")}',"
        "f'ROUND11_DYNAMIC_TIMING_ENABLE={int(a.placement==\"request_boundary\")}',"
        "'ROUND11_B_ATTRIBUTION_ENABLE=1',"
        "'ROUND11_B_SPANS=/out/round11-b-spans.jsonl',"
        "'ROUND11_B_BINDINGS=/out/round11-b-bindings.json'):args+=['-e',env]"
    )
    source = replace_once(source, old_env, new_env, "T5 observer environment")
    source = replace_once(
        source,
        "if budget.get('stage')!='ROUND11_T4_PROCESS':raise RuntimeError('wrong Round11 T4 budget')",
        "if budget.get('stage')!='ROUND11_T5_PROCESS':raise RuntimeError('wrong Round11 T5 budget')",
        "T5 launcher budget",
    )
    compile(source, "launch_hybrid.py", "exec")
    return source


def refresh_overlay(root: Path, runtime_source: Path) -> None:
    overlay = root / "overlay-python"
    target = overlay / "sglang/srt/layers/round9/round11_b_attribution.py"
    shutil.copy2(runtime_source, target)
    scheduler = overlay / "sglang/srt/managers/scheduler.py"
    scheduler.write_text(patch_scheduler(scheduler.read_text()))
    compile(scheduler.read_text(), str(scheduler), "exec")

    tools = root / "source/tools"
    sys.path.insert(0, str(tools))
    try:
        from install_candidate_overlay import tree_digest, tree_files

        receipt_path = overlay / "round9-candidate-overlay.json"
        receipt = json.loads(receipt_path.read_text())
        receipt.setdefault("modules", {})[runtime_source.name] = sha256(runtime_source)
        receipt["candidate_files"] = tree_files(overlay)
        receipt["candidate_tree_sha256"] = tree_digest(receipt["candidate_files"])
        receipt["round11_b_attribution"] = {
            "status": "INSTALLED_BOUNDED_OBSERVER_NOT_UNPROFILED_PERFORMANCE",
            "runtime_sha256": sha256(runtime_source),
            "scheduler_sha256": sha256(scheduler),
            "device_query": "DEFERRED_QUERY_ONLY_NO_SYNCHRONIZE",
        }
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    finally:
        sys.path.remove(str(tools))


def source_manifest(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and "__pycache__" not in path.parts
        and str(path.relative_to(root)) not in MUTABLE
        and not str(path.relative_to(root)).startswith("host-controller")
    }


def controller_source(*, block: str, version: int) -> str:
    name = f"kt-round11-t5-hotspot-v{version}"
    return f'''"""Launch immutable Round11 T5 attribution block {block}."""
import argparse,hashlib,json,pathlib,subprocess,time,traceback
from m3_cleanup import wait_for_guard_exit
p=argparse.ArgumentParser();p.add_argument('--admission',required=True);a=p.parse_args()
root=pathlib.Path(__file__).resolve().parent;base=pathlib.Path('/mnt/data1/kt-round9')
out=base/'evidence'/'20260926-round11-t5-v{version}';name='{name}'
claim=root/'host-controller.json';manifest=json.loads((root/'round11-source-sha256.json').read_text())
excluded={{'round11-source-sha256.json','execution-budget.json','execution-budget.lock','host-controller.json'}}
actual={{str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest() for path in root.rglob('*') if path.is_file() and '__pycache__' not in path.parts and str(path.relative_to(root)) not in excluded and not str(path.relative_to(root)).startswith('host-controller')}}
if actual!=manifest:raise RuntimeError('Round11 T5 snapshot manifest mismatch')
with claim.open('x') as stream:json.dump(dict(block='{block}',version={version},started=time.time(),status='LAUNCHING'),stream)
args=['python3',str(root/'source/tools/launch_hybrid.py'),'--version','{version}',
 '--storage','MXFP4','--placement','static','--resource-mode','shared',
 '--shared-training-container','{TRAINING_ID}','--bank-receipt',str(root/'rebound/bank-receipt.json'),
 '--device-gate',str(base/'evidence/20260920-c128-expert-v19/result.json'),
 '--overlay-python',str(root/'overlay-python'),'--framework-profile',str(root/'framework-profile.json'),
 '--context-target','4096','--context-reference-profile',str(base/'evidence/20260922-plan10-m3-baseline-v64/context-reference-profile.json'),
 '--streaming-profile',str(root/'streaming-profile.json'),'--cancel-trace','--cancel-closure',
 '--cost-mode','OFF','--cost-manifest',str(root/'cost-manifest.json'),
 '--cost-budget',str(root/'execution-budget.json'),'--cost-admission',a.admission]
report=dict(block='{block}',version={version},launch_command=args,status='LAUNCHING',started=time.time())
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
 supervisor=['python3',str(root/'round11_t5_supervisor.py'),'--version','{version}','--block','{block}',
  '--container',start['name'],'--container-id',start['container_id'],'--out',str(out),
  '--driver-root','/round9/'+str(root.relative_to(base))]
 subprocess.run(supervisor,check=True,timeout=21780)
 report['status']=json.loads((out/'round11-t5-supervisor.json').read_text())['status']
except BaseException as error:
 report.update(status='FAIL',error=repr(error),traceback=traceback.format_exc())
 try:report['failed_launch_cleanup']=cleanup_failed_launch()
 except BaseException as cleanup:report['cleanup_error']=repr(cleanup)
 raise
finally:report['ended']=time.time();save()
'''


def supervisor_source(*, block: str, version: int) -> str:
    name = f"kt-round11-t5-hotspot-v{version}"
    return f'''"""Supervise immutable Round11 T5 attribution block {block}."""
import argparse,json,pathlib,subprocess,time,traceback
from m3_cleanup import wait_for_guard_exit
p=argparse.ArgumentParser();p.add_argument('--version',type=int,required=True);p.add_argument('--block',required=True);p.add_argument('--container',required=True);p.add_argument('--container-id',required=True);p.add_argument('--out',type=pathlib.Path,required=True);p.add_argument('--driver-root',required=True);a=p.parse_args()
if (a.version,a.block,a.container)!=({version},'{block}','{name}') or len(a.container_id)!=64:raise ValueError('fixed Round11 T5 identity required')
def command(args,timeout=30):return subprocess.check_output(args,text=True,timeout=timeout)
def inspect():
 document=json.loads(command(['docker','inspect',a.container]))[0]
 if document['Id']!=a.container_id:raise RuntimeError('owned container identity changed')
 return document
report={{'status':'WAITING_FOR_SERVER','version':a.version,'block':a.block,'container_id':a.container_id,'started_at':time.time(),'generation_driver_starts':0}}
path=a.out/'round11-t5-supervisor.json';path.touch(exist_ok=False)
def save():path.write_text(json.dumps(report,indent=2)+'\\n')
save();child=None;raw=None
try:
 deadline=time.time()+21600
 while time.time()<deadline:
  document=inspect()
  if not document['State']['Running']:raise RuntimeError('server exited before T5 completion')
  if any(file.stat().st_size>512*1024**2 for file in a.out.glob('*.log')):raise RuntimeError('log ceiling exceeded')
  log=a.out/'server.log'
  if child is None and log.exists():
   with log.open('rb') as stream:stream.seek(max(0,log.stat().st_size-65536));tail=stream.read().decode(errors='replace')
   if 'Uvicorn running on http://127.0.0.1:30091' in tail or 'The server is fired up and ready to roll!' in tail:
    if (a.out/'round11-t5-process-raw.json').exists():raise RuntimeError('T5 ledger already exists; no retry')
    args=['docker','exec',a.container_id,'python',a.driver_root+'/round11_t5_driver.py','--block',a.block,'--out','/out','--fixture',a.driver_root+'/round10-t1-fixture.json','--parent-identity','{PARENT_IDENTITY}']
    report.update(status='RUNNING_T5',generation_driver_starts=1,driver_command=args);save()
    with (a.out/'round11-t5-driver.log').open('x') as stream:child=subprocess.Popen(args,stdout=stream,stderr=subprocess.STDOUT)
  if child is not None and child.poll() is not None:
   report['driver_exit']=child.returncode;raw_path=a.out/'round11-t5-process-raw.json'
   if not raw_path.exists():raise RuntimeError('T5 driver exited before publishing raw result')
   raw=json.loads(raw_path.read_text())
   if raw.get('generation_attempts')!=4 or raw.get('automatic_retries')!=0:raise RuntimeError('T5 result budget mismatch')
   report['status']='DRIVER_PASS_WAITING_CLEANUP' if child.returncode==0 and raw.get('status')=='PASS_ROUND11_T5_DRIVER_RAW' else 'FAIL'
   break
  time.sleep(5)
 else:raise TimeoutError('bounded T5 supervisor deadline')
except BaseException as error:report.update(status='FAIL',error=repr(error),traceback=traceback.format_exc())
finally:
 report['ended_at']=time.time();save()
 try:
  document=inspect();stop_started=time.time();was_running=document['State']['Running']
  if was_running:subprocess.run(['docker','stop','-t','10',a.container_id],check=True,timeout=30)
  report['owned_exit_state']=inspect()['State'];report['guard_exit_ack']=wait_for_guard_exit(a.out/'monitor.jsonl',not_before=stop_started if was_running else report['started_at'])
  subprocess.run(['docker','rm',a.container_id],check=True,timeout=30);report['owned_container_removed']=True
  proc=command(['npu-smi','info','-t','proc-mem','-i','6','-c','0']);(a.out/'post-cleanup-proc-mem.txt').write_text(proc)
  if 'No process in device' not in proc:raise RuntimeError('physical12 is not empty after T5 cleanup')
  report['physical12_after']='EMPTY'
 except BaseException as cleanup:report['cleanup_error']=repr(cleanup);report['status']='FAIL_CLEANUP'
 if child is not None and child.poll() is None:
  try:child.wait(timeout=20)
  except subprocess.TimeoutExpired:child.terminate()
 if report['status']=='DRIVER_PASS_WAITING_CLEANUP' and raw is not None:
  final=dict(raw);final['status']='PASS_ROUND11_T5_PROCESS';final['cleanup']={{'guard_exit_ack':True,'container_removed':True,'physical12_after':'EMPTY'}}
  (a.out/'round11-t5-final.json').write_text(json.dumps(final,indent=2)+'\\n');report['status']='PASS_ROUND11_T5_PROCESS'
 save()
if report['status']!='PASS_ROUND11_T5_PROCESS':raise SystemExit(1)
'''


def build(args) -> None:
    block = args.block.upper()
    if args.version not in ROLE_CONTRACT.get(block, ()):
        raise ValueError("T5 block/version identity mismatch")
    if args.output.exists():
        raise FileExistsError(args.output)
    shutil.copytree(args.parent, args.output, ignore=shutil.ignore_patterns("__pycache__"))
    # Frozen parents are intentionally read-only.  Restore owner write access
    # only inside the newly copied build tree; the caller freezes the finished
    # snapshot again after its manifest has been verified.
    for path in (args.output, *args.output.rglob("*")):
        path.chmod(path.stat().st_mode | stat.S_IWUSR)
    for name in ("controller-F.json", "execution-budget.lock", "host-controller.json"):
        path = args.output / name
        if path.exists():
            path.unlink()

    local = Path(__file__).resolve().parents[2]
    runtime_source = local / "migration_bridge/round9/runtime/round11_b_attribution.py"
    shutil.copy2(runtime_source, args.output / "source/runtime/round11_b_attribution.py")
    refresh_overlay(args.output, runtime_source)

    launcher = args.output / "source/tools/launch_hybrid.py"
    launcher.write_text(patch_launcher(launcher.read_text(), block=block, version=args.version))
    shutil.copy2(local / "artifacts/round11/round11_t5_driver.py", args.output / "round11_t5_driver.py")
    shutil.copy2(local / "artifacts/round11/round11_monitor_owned.py", args.output / "source/tools/monitor_owned_server.py")
    (args.output / "host_controller.py").write_text(controller_source(block=block, version=args.version))
    (args.output / "round11_t5_supervisor.py").write_text(supervisor_source(block=block, version=args.version))
    compile((args.output / "host_controller.py").read_text(), "host_controller.py", "exec")
    compile((args.output / "round11_t5_supervisor.py").read_text(), "round11_t5_supervisor.py", "exec")

    framework = json.loads((args.output / "framework-profile.json").read_text())
    if framework["prefill"]["update_residents"] is not False:
        raise ValueError("T5 requires HFINAL Static/Dynamic-OFF parent")
    fixture = json.loads((args.output / "round10-t1-fixture.json").read_text())
    for case_id, length in (("edge128", 128), ("canary_math391", 27)):
        values = fixture.get("cases", {}).get(case_id, {}).get("input_ids")
        if not isinstance(values, list) or len(values) != length:
            raise ValueError(f"T5 fixture {case_id} identity mismatch")

    budget = {
        "schema": 1,
        "stage": "ROUND11_T5_PROCESS",
        "version": args.version,
        "block": block,
        "model_starts_limit": 1,
        "model_starts_used": 0,
        "starts": [],
        "automatic_retries_limit": 0,
    }
    (args.output / "execution-budget.json").write_text(json.dumps(budget, indent=2) + "\n")
    contract = {
        "schema": 1,
        "status": "FROZEN_NOT_RUN",
        "stage": "ROUND11_T5_GDN_QSA_PLE_HOTSPOT_RANKING",
        "version": args.version,
        "block": block,
        "parent_identity": PARENT_IDENTITY,
        "dynamic": "OFF",
        "threshold_tokens": 129,
        "profiler": "BOUNDED_OBSERVER_ON",
        "generation_limit": 4,
        "measurement_requests": 2,
        "automatic_retries": 0,
        "observer": {
            "gdn_layers": [0, 24, 46],
            "qsa_layers": [3, 23, 47],
            "ple_layers": [1],
            "device_query": "DEFERRED_QUERY_ONLY_NO_SYNCHRONIZE",
        },
    }
    (args.output / "ROUND11_T5_SNAPSHOT_CONTRACT.json").write_text(
        json.dumps(contract, indent=2) + "\n"
    )
    manifest = source_manifest(args.output)
    (args.output / "round11-source-sha256.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print("PASS_ROUND11_T5_SNAPSHOT", block, args.version, len(manifest), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block", choices=("A", "B"), required=True)
    parser.add_argument("--version", type=int, required=True)
    args = parser.parse_args()
    build(args)


if __name__ == "__main__":
    main()
