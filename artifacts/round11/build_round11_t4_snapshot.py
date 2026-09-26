"""Build one immutable Round11 Dynamic break-even model snapshot."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import sys


TRAINING_ID = "0419bcfee46f317d6833e038c28e81cf3d3ecf47323a423062bac2764e0ab2ef"
PARENT_IDENTITY = "ROUND11_HFINAL_V107_ORIGINAL_C128_MXFP4"
QUALIFIED_DYNAMIC_SHA256 = "a2db99532a9faee15c3cf899579b9018601156e077ab61908ca0c938d8092114"
INSTRUMENTED_DYNAMIC_SHA256 = "485fa00dfce0f072ec6d75578724393aaa5bccc68ce60a99ded79d8018a91e34"
TIMING_DIFF_SHA256 = "aa1a7df1509bf3c5780e9ccf17fed44c285bd9cd5c5843c69d0ba8b1694427bf"
HFINAL_BANK_RECEIPT_SHA256 = "e6d9dc9d3d7753fe26721c97e9f62575d67cb48af0b2ec957ac2ddb369450482"
STATIC_PROFILE_CANONICAL_SHA256 = "65f9743ff9ef39a9314594d4f82c27d45dafe86ea0c1e46d0790312b2b2fade2"
DYNAMIC_PROFILE_CANONICAL_SHA256 = "7e205683c583f9eb1f0f3cd99dad6e1479958f2fc3738ed9d8fb32ecf19c79b1"
BANK_PAYLOAD_SHA256 = "1bc5b1e00bc0722fefa8aab05799ac8d2b48b572b8dd57947e5972fde4d6e1e2"
ROUND10_DYNAMIC_RESULT_SHA256 = "66680be7e81ae6777363723944b8dfb7b7730d144dd49d1ebafd92e3f7abe897"
ROUND10_DYNAMIC_BANK_RECEIPT_SHA256 = "32b5e357a270931ec05f2487b6a505c90e7920eadc4b23c89e9672283db38de1"
MUTABLE = {
    "round11-source-sha256.json",
    "execution-budget.json",
    "execution-budget.lock",
    "host-controller.json",
}
RUNTIME_FILES = (
    "dynamic.py",
    "prefill_binding.py",
    "prefill_end_boundary.py",
    "dynamic_boundary.py",
    "cancel_lifecycle.py",
)
ROLE_CONTRACT = {
    ("STATIC", "A"): (120, 124),
    ("DYNAMIC", "A"): (121, 125, 128, 132, 134),
    ("DYNAMIC", "B"): (122, 126, 129, 133, 135),
    ("STATIC", "B"): (123, 127),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _replace_once(source: str, old: str, new: str, label: str) -> str:
    if source.count(old) != 1:
        raise ValueError(f"{label} precondition mismatch")
    return source.replace(old, new)


def build_timing_source_proof(qualified: Path, instrumented: Path) -> dict:
    qualified_hash = sha256(qualified)
    instrumented_hash = sha256(instrumented)
    diff = "".join(
        difflib.unified_diff(
            qualified.read_text().splitlines(keepends=True),
            instrumented.read_text().splitlines(keepends=True),
            fromfile="qualified/dynamic.py",
            tofile="instrumented/dynamic.py",
        )
    )
    diff_hash = hashlib.sha256(diff.encode()).hexdigest()
    if (
        qualified_hash != QUALIFIED_DYNAMIC_SHA256
        or instrumented_hash != INSTRUMENTED_DYNAMIC_SHA256
        or diff_hash != TIMING_DIFF_SHA256
    ):
        raise ValueError("Round11 Dynamic timing source derivation drifted")
    return {
        "schema": 1,
        "status": "PASS_ROUND11_TIMING_ONLY_SOURCE_DERIVATION",
        "qualified_dynamic_sha256": qualified_hash,
        "instrumented_dynamic_sha256": instrumented_hash,
        "unified_diff_sha256": diff_hash,
        "semantic_change": "MONOTONIC_TIMING_OBSERVER_ONLY",
    }


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def rebind_dynamic_candidate_profile(root: Path) -> dict:
    profile_path = root / "rebound/candidate-profile.json"
    receipt_path = root / "rebound/bank-receipt.json"
    profile = json.loads(profile_path.read_text())
    receipt = json.loads(receipt_path.read_text())
    if (
        profile != {
            "workers": 128,
            "storage": "MXFP4",
            "placement": "static",
            "numa_nodes": [2, 3, 4, 7],
        }
        or _canonical_sha256(profile) != STATIC_PROFILE_CANONICAL_SHA256
        or sha256(receipt_path) != HFINAL_BANK_RECEIPT_SHA256
        or _canonical_sha256(receipt.get("banks")) != BANK_PAYLOAD_SHA256
        or receipt.get("status") != "VERIFIED48_BANK_BYTES_TYPES_SOURCE_NOT_MODEL_QUALIFIED"
    ):
        raise ValueError("HFINAL Dynamic profile rebind precondition drifted")
    profile["placement"] = "request_boundary"
    dynamic_hash = _canonical_sha256(profile)
    if dynamic_hash != DYNAMIC_PROFILE_CANONICAL_SHA256:
        raise ValueError("Dynamic candidate profile identity drifted")
    proof = {
        "schema": 1,
        "status": "PASS_DYNAMIC_PROFILE_ONLY_REBIND",
        "parent_bank_receipt_sha256": HFINAL_BANK_RECEIPT_SHA256,
        "parent_profile_sha256": STATIC_PROFILE_CANONICAL_SHA256,
        "dynamic_profile_sha256": dynamic_hash,
        "bank_payload_sha256": BANK_PAYLOAD_SHA256,
        "round10_dynamic_result_sha256": ROUND10_DYNAMIC_RESULT_SHA256,
        "round10_dynamic_bank_receipt_sha256": ROUND10_DYNAMIC_BANK_RECEIPT_SHA256,
        "changed_field": "placement:static->request_boundary",
    }
    receipt["profile_sha256"] = dynamic_hash
    receipt["round11_dynamic_profile_rebind"] = proof
    profile_path.write_text(json.dumps(profile, indent=2) + "\n")
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    return proof


def role_name(mode: str, block: str, version: int) -> str:
    allowed = ROLE_CONTRACT.get((mode, block), ())
    if version not in allowed:
        raise ValueError("T4 mode/block/version identity mismatch")
    return f"kt-round11-t4-{mode.lower()}-v{version}"


def patch_launcher(source: str, *, mode: str, block: str, version: int) -> str:
    name = role_name(mode, block, version)
    source = _replace_once(
        source,
        "import argparse,hashlib,json,pathlib,re,shutil,subprocess,time",
        "import argparse,difflib,hashlib,json,pathlib,re,shutil,subprocess,time",
        "launcher timing-proof import",
    )
    source = _replace_once(
        source,
        "if a.version != 107 or a.cost_mode != 'OFF':",
        f"if a.version != {version} or a.cost_mode != 'OFF':",
        "launcher version",
    )
    source = _replace_once(
        source,
        "name='kt-round10-t6-final-v107';out=root/'evidence'/'20260926-round10-t6-v107'",
        f"name='{name}';out=root/'evidence'/'20260926-round11-t4-v{version}'",
        "launcher output identity",
    )
    source = _replace_once(
        source,
        "if budget.get('stage')!='ROUND10_T6_FINAL':raise RuntimeError('wrong Round10 T6 budget')",
        "if budget.get('stage')!='ROUND11_T4_PROCESS':raise RuntimeError('wrong Round11 T4 budget')",
        "launcher budget stage",
    )
    source = _replace_once(
        source,
        "'ROUND9_DYNAMIC_EVENTS=/out/dynamic-events.jsonl',f'ROUND9_DYNAMIC_BOUNDARY_ENABLE={int(a.placement==\"request_boundary\")}'):args+=['-e',env]",
        "'ROUND9_DYNAMIC_EVENTS=/out/dynamic-events.jsonl',f'ROUND9_DYNAMIC_BOUNDARY_ENABLE={int(a.placement==\"request_boundary\")}',f'ROUND11_DYNAMIC_TIMING_ENABLE={int(a.placement==\"request_boundary\")}'):args+=['-e',env]",
        "Round11 Dynamic timing environment",
    )
    source = _replace_once(
        source,
        """if a.placement=='request_boundary':
 gate_source=json.loads((a.device_gate.parent/'source-sha256.json').read_text())
 for module in ('dynamic_adapter.py','dynamic.py','expert_source.py'):
  if digest(sglang/'sglang/srt/layers/round9'/module)!=gate_source['runtime/'+module]:raise RuntimeError('Dynamic math adapter differs from qualified device gate')""",
        """if a.placement=='request_boundary':
 gate_source=json.loads((a.device_gate.parent/'source-sha256.json').read_text())
 for module in ('dynamic_adapter.py','expert_source.py'):
  if digest(sglang/'sglang/srt/layers/round9'/module)!=gate_source['runtime/'+module]:raise RuntimeError('Dynamic math adapter differs from qualified device gate')
 profile_proof=json.loads((source.parent/'ROUND11_DYNAMIC_PROFILE_REBIND.json').read_text())
 if receipt.get('round11_dynamic_profile_rebind')!=profile_proof or profile_proof.get('status')!='PASS_DYNAMIC_PROFILE_ONLY_REBIND' or profile_proof.get('dynamic_profile_sha256')!=candidate_sha:raise RuntimeError('Round11 Dynamic profile rebind proof mismatch')
 proof_path=source.parent/'ROUND11_DYNAMIC_TIMING_SOURCE_PROOF.json';qualified_dynamic=source.parent/'round11-qualified-dynamic.py';instrumented_dynamic=sglang/'sglang/srt/layers/round9/dynamic.py'
 dynamic_diff=''.join(difflib.unified_diff(qualified_dynamic.read_text().splitlines(keepends=True),instrumented_dynamic.read_text().splitlines(keepends=True),fromfile='qualified/dynamic.py',tofile='instrumented/dynamic.py'))
 expected_proof={'schema':1,'status':'PASS_ROUND11_TIMING_ONLY_SOURCE_DERIVATION','qualified_dynamic_sha256':gate_source['runtime/dynamic.py'],'instrumented_dynamic_sha256':digest(instrumented_dynamic),'unified_diff_sha256':hashlib.sha256(dynamic_diff.encode()).hexdigest(),'semantic_change':'MONOTONIC_TIMING_OBSERVER_ONLY'}
 if digest(qualified_dynamic)!=gate_source['runtime/dynamic.py'] or json.loads(proof_path.read_text())!=expected_proof:raise RuntimeError('Round11 Dynamic timing derivation proof mismatch')""",
        "Round11 Dynamic timing source proof",
    )
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


def refresh_overlay(root: Path) -> None:
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
        receipt["round11_dynamic_break_even"] = {
            "status": "INSTALLED_NOT_FULL_MODEL_QUALIFIED",
            "runtime_files": {name: sha256(root / "source/runtime" / name) for name in RUNTIME_FILES},
        }
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    finally:
        sys.path.remove(str(tools))


def controller_source(*, mode: str, block: str, version: int) -> str:
    name = role_name(mode, block, version)
    placement = "request_boundary" if mode == "DYNAMIC" else "static"
    return f'''"""Launch one immutable Round11 T4 {mode}-{block} model process."""
import argparse,hashlib,json,pathlib,subprocess,time,traceback
from m3_cleanup import wait_for_guard_exit
p=argparse.ArgumentParser();p.add_argument('--admission',required=True);a=p.parse_args()
root=pathlib.Path(__file__).resolve().parent;base=pathlib.Path('/mnt/data1/kt-round9')
out=base/'evidence'/'20260926-round11-t4-v{version}';name='{name}'
claim=root/'host-controller.json';manifest=json.loads((root/'round11-source-sha256.json').read_text())
excluded={{'round11-source-sha256.json','execution-budget.json','execution-budget.lock','host-controller.json'}}
actual={{str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest() for path in root.rglob('*') if path.is_file() and '__pycache__' not in path.parts and str(path.relative_to(root)) not in excluded and not str(path.relative_to(root)).startswith('host-controller')}}
if actual!=manifest:raise RuntimeError('Round11 T4 snapshot manifest mismatch')
with claim.open('x') as stream:json.dump(dict(mode='{mode}',block='{block}',version={version},started=time.time(),status='LAUNCHING'),stream)
args=['python3',str(root/'source/tools/launch_hybrid.py'),'--version','{version}',
 '--storage','MXFP4','--placement','{placement}','--resource-mode','shared',
 '--shared-training-container','{TRAINING_ID}','--bank-receipt',str(root/'rebound/bank-receipt.json'),
 '--device-gate',str(base/'evidence/20260920-c128-expert-v19/result.json'),
 '--overlay-python',str(root/'overlay-python'),'--framework-profile',str(root/'framework-profile.json'),
 '--context-target','4096','--context-reference-profile',str(base/'evidence/20260922-plan10-m3-baseline-v64/context-reference-profile.json'),
 '--streaming-profile',str(root/'streaming-profile.json'),'--cancel-trace','--cancel-closure',
 '--cost-mode','OFF','--cost-manifest',str(root/'cost-manifest.json'),
 '--cost-budget',str(root/'execution-budget.json'),'--cost-admission',a.admission]
report=dict(mode='{mode}',block='{block}',version={version},launch_command=args,status='LAUNCHING',started=time.time())
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
 supervisor=['python3',str(root/'round11_t4_supervisor.py'),'--version','{version}','--mode','{mode}','--block','{block}',
  '--container',start['name'],'--container-id',start['container_id'],'--out',str(out),
  '--driver-root','/round9/'+str(root.relative_to(base))]
 subprocess.run(supervisor,check=True,timeout=21780)
 report['status']=json.loads((out/'round11-t4-supervisor.json').read_text())['status']
except BaseException as error:
 report.update(status='FAIL',error=repr(error),traceback=traceback.format_exc())
 try:report['failed_launch_cleanup']=cleanup_failed_launch()
 except BaseException as cleanup:report['cleanup_error']=repr(cleanup)
 raise
finally:report['ended']=time.time();save()
'''


def supervisor_source(*, mode: str, block: str, version: int) -> str:
    name = role_name(mode, block, version)
    return f'''"""Supervise one immutable Round11 T4 {mode}-{block} process."""
import argparse,json,pathlib,subprocess,time,traceback
from m3_cleanup import wait_for_guard_exit
p=argparse.ArgumentParser();p.add_argument('--version',type=int,required=True);p.add_argument('--mode',required=True);p.add_argument('--block',required=True);p.add_argument('--container',required=True);p.add_argument('--container-id',required=True);p.add_argument('--out',type=pathlib.Path,required=True);p.add_argument('--driver-root',required=True);a=p.parse_args()
if (a.version,a.mode,a.block,a.container)!=({version},'{mode}','{block}','{name}') or len(a.container_id)!=64:raise ValueError('fixed Round11 T4 identity required')
def command(args,timeout=30):return subprocess.check_output(args,text=True,timeout=timeout)
def inspect():
 document=json.loads(command(['docker','inspect',a.container]))[0]
 if document['Id']!=a.container_id:raise RuntimeError('owned container identity changed')
 return document
report={{'status':'WAITING_FOR_SERVER','version':a.version,'mode':a.mode,'block':a.block,'container_id':a.container_id,'started_at':time.time(),'generation_driver_starts':0}}
path=a.out/'round11-t4-supervisor.json';path.touch(exist_ok=False)
def save():path.write_text(json.dumps(report,indent=2)+'\\n')
save();child=None;raw=None
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
    if (a.out/'round11-t4-process.json').exists():raise RuntimeError('T4 ledger already exists; no retry')
    args=['docker','exec',a.container_id,'python',a.driver_root+'/round11_t4_driver.py','--mode',a.mode,'--block',a.block,'--out','/out','--fixture',a.driver_root+'/round10-t1-fixture.json','--parent-identity','{PARENT_IDENTITY}']
    report.update(status='RUNNING_T4',generation_driver_starts=1,driver_command=args);save()
    with (a.out/'round11-t4-driver.log').open('x') as stream:child=subprocess.Popen(args,stdout=stream,stderr=subprocess.STDOUT)
  if child is not None and child.poll() is not None:
   report['driver_exit']=child.returncode;raw_path=a.out/'round11-t4-process.json'
   if not raw_path.exists():raise RuntimeError('T4 driver exited before publishing raw result')
   raw=json.loads(raw_path.read_text())
   if raw.get('generation_attempts')!=5 or raw.get('automatic_retries')!=0:raise RuntimeError('T4 result budget mismatch')
   report['status']='DRIVER_PASS_WAITING_CLEANUP' if child.returncode==0 and raw.get('status')=='PASS_ROUND11_T4_DRIVER_RAW' else 'FAIL'
   break
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
  proc=command(['npu-smi','info','-t','proc-mem','-i','6','-c','0']);(a.out/'post-cleanup-proc-mem.txt').write_text(proc)
  if 'No process in device' not in proc:raise RuntimeError('physical12 is not empty after T4 cleanup')
  report['physical12_after']='EMPTY'
 except BaseException as cleanup:report['cleanup_error']=repr(cleanup);report['status']='FAIL_CLEANUP'
 if child is not None and child.poll() is None:
  try:child.wait(timeout=20)
  except subprocess.TimeoutExpired:child.terminate()
 if report['status']=='DRIVER_PASS_WAITING_CLEANUP' and raw is not None:
  final=dict(raw);final['status']='PASS_ROUND11_T4_PROCESS';final['cleanup']={{'guard_exit_ack':True,'container_removed':True,'physical12_after':'EMPTY'}}
  (a.out/'round11-t4-final.json').write_text(json.dumps(final,indent=2)+'\\n');report['status']='PASS_ROUND11_T4_PROCESS'
 save()
if report['status']!='PASS_ROUND11_T4_PROCESS':raise SystemExit(1)
'''


def build(args) -> None:
    mode = args.mode.upper()
    block = args.block.upper()
    role_name(mode, block, args.version)
    if args.output.exists():
        raise FileExistsError(args.output)
    shutil.copytree(args.parent, args.output, ignore=shutil.ignore_patterns("__pycache__"))
    for name in ("controller-F.json", "execution-budget.lock", "host-controller.json"):
        path = args.output / name
        if path.exists():
            path.unlink()
    if mode == "DYNAMIC":
        profile_proof = rebind_dynamic_candidate_profile(args.output)
        (args.output / "ROUND11_DYNAMIC_PROFILE_REBIND.json").write_text(
            json.dumps(profile_proof, indent=2, sort_keys=True) + "\n"
        )
    launcher = args.output / "source/tools/launch_hybrid.py"
    launcher.write_text(patch_launcher(launcher.read_text(), mode=mode, block=block, version=args.version))
    local = Path(__file__).resolve().parents[2]
    if mode == "DYNAMIC":
        qualified = args.output / "source/runtime/dynamic.py"
        instrumented = local / "migration_bridge/round9/runtime/dynamic.py"
        shutil.copy2(qualified, args.output / "round11-qualified-dynamic.py")
        proof = build_timing_source_proof(qualified, instrumented)
        (args.output / "ROUND11_DYNAMIC_TIMING_SOURCE_PROOF.json").write_text(
            json.dumps(proof, indent=2, sort_keys=True) + "\n"
        )
    for name in RUNTIME_FILES:
        shutil.copy2(local / "migration_bridge/round9/runtime" / name, args.output / "source/runtime" / name)
    shutil.copy2(local / "artifacts/round11/round11_monitor_owned.py", args.output / "source/tools/monitor_owned_server.py")
    shutil.copy2(local / "artifacts/round11/round11_t4_driver.py", args.output / "round11_t4_driver.py")
    fixture_path = args.output / "round10-t1-fixture.json"
    fixture = json.loads(fixture_path.read_text())
    ids = fixture.get("cases", {}).get("edge64", {}).get("input_ids")
    if not isinstance(ids, list) or len(ids) != 64 or any(type(value) is not int for value in ids):
        raise ValueError("HFINAL parent is missing the exact Round10 64-token fixture")
    framework_path = args.output / "framework-profile.json"
    framework = json.loads(framework_path.read_text())
    framework["prefill"]["update_residents"] = mode == "DYNAMIC"
    framework_path.write_text(json.dumps(framework, indent=2) + "\n")
    refresh_overlay(args.output)
    (args.output / "host_controller.py").write_text(controller_source(mode=mode, block=block, version=args.version))
    (args.output / "round11_t4_supervisor.py").write_text(supervisor_source(mode=mode, block=block, version=args.version))
    budget = {
        "schema": 1,
        "stage": "ROUND11_T4_PROCESS",
        "version": args.version,
        "mode": mode,
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
        "stage": "ROUND11_T4_DYNAMIC_BREAK_EVEN",
        "version": args.version,
        "mode": mode,
        "block": block,
        "parent_identity": PARENT_IDENTITY,
        "dynamic": "PREFILL_END_ON" if mode == "DYNAMIC" else "OFF",
        "threshold_tokens": 129,
        "generation_limit": 5,
        "automatic_retries": 0,
        "only_variable": "PREFILL_END_DYNAMIC_ENABLE",
    }
    (args.output / "ROUND11_T4_SNAPSHOT_CONTRACT.json").write_text(json.dumps(contract, indent=2) + "\n")
    manifest = source_manifest(args.output)
    (args.output / "round11-source-sha256.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print("PASS_ROUND11_T4_SNAPSHOT", mode, block, args.version, len(manifest), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("STATIC", "DYNAMIC"), required=True)
    parser.add_argument("--block", choices=("A", "B"), required=True)
    parser.add_argument("--version", type=int, required=True)
    args = parser.parse_args()
    build(args)


if __name__ == "__main__":
    main()
