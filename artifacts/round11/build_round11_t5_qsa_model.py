#!/usr/bin/env python3
"""Build immutable v145 for the single Round11 QSA full-model candidate block."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import stat
import sys


VERSION = 145
SUPPORTED_VERSIONS = (145, 146)
TRAINING_ID = "0419bcfee46f317d6833e038c28e81cf3d3ecf47323a423062bac2764e0ab2ef"
PARENT_IDENTITY = "ROUND11_HFINAL_V107_ORIGINAL_C128_MXFP4"
COMPONENT_ANALYSIS_SHA256 = "83ea9975210eed86ac487d8f0010afa2b663768b2fa2bef4930931a3e9dbd787"
COMPONENT_RESULT_SHA256 = "651fd396942b61bab45b73d34c6dcb66399a7690753e3617eac44fe7f92f0465"
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


def patch_launcher(source: str, *, version: int = VERSION) -> str:
    if version not in SUPPORTED_VERSIONS:
        raise ValueError("unsupported QSA model snapshot version")
    source = replace_once(
        source,
        "if a.version != 127 or a.cost_mode != 'OFF':",
        f"if a.version != {version} or a.cost_mode != 'OFF':",
        "QSA model launcher version",
    )
    source = replace_once(
        source,
        "name='kt-round11-t4-static-v127';out=root/'evidence'/'20260926-round11-t4-v127'",
        f"name='kt-round11-t5-qsa-candidate-v{version}';out=root/'evidence'/'20260926-round11-t5-qsa-model-v{version}'",
        "QSA model launcher identity",
    )
    source = replace_once(
        source,
        "if budget.get('stage')!='ROUND11_T4_PROCESS':raise RuntimeError('wrong Round11 T4 budget')",
        "if budget.get('stage')!='ROUND11_T5_QSA_MODEL':raise RuntimeError('wrong Round11 T5 QSA model budget')",
        "QSA model launcher budget",
    )
    compile(source, "launch_hybrid.py", "exec")
    return source


def patch_context_gate(source: str) -> str:
    old = """ require_context_reference_gate(context_path,spec['sha256'],source_identity=receipt['source_identity'],
  source_sha256={module:digest(sglang/'sglang/srt/layers/round9'/module) for module in SOURCE_FILES},
  torch_version=context_receipt.get('torch_version'),torch_npu_version=context_receipt.get('torch_npu_version'))"""
    new = """ qsa_proof=json.loads((source.parent/'ROUND11_QSA_GATE_REBIND.json').read_text())
 qualified_qsa=source.parent/'round11-qualified-qsa.py'
 base_source_sha256={module:digest(sglang/'sglang/srt/layers/round9'/module) for module in SOURCE_FILES}
 base_source_sha256['qsa.py']=digest(qualified_qsa)
 require_context_reference_gate(context_path,spec['sha256'],source_identity=receipt['source_identity'],
  source_sha256=base_source_sha256,
  torch_version=context_receipt.get('torch_version'),torch_npu_version=context_receipt.get('torch_npu_version'))
 qsa_analysis=json.loads((source.parent/'ROUND11_T5_QSA_COMPONENT_V144_ANALYSIS.json').read_text())
 qsa_component=json.loads((source.parent/'ROUND11_T5_QSA_COMPONENT_V144_RESULT.json').read_text())
 if (qsa_proof.get('status')!='PASS_ROUND11_QSA_GATE_REBIND' or
     qsa_proof.get('qualified_qsa_sha256')!=digest(qualified_qsa) or
     qsa_proof.get('candidate_qsa_sha256')!=digest(sglang/'sglang/srt/layers/round9/qsa.py') or
     qsa_proof.get('component_analysis_sha256')!=digest(source.parent/'ROUND11_T5_QSA_COMPONENT_V144_ANALYSIS.json') or
     qsa_proof.get('component_result_sha256')!=digest(source.parent/'ROUND11_T5_QSA_COMPONENT_V144_RESULT.json') or
     qsa_analysis.get('disposition')!='QSA_COMPONENT_PROMOTE_TO_REQUEST' or
     qsa_component.get('status')!='PASS_ROUND11_T5_QSA_COMPONENT' or
     qsa_component.get('candidate')!='QUERY_AWARE_VECTORIZED_SPARSE_GQA' or
     qsa_component.get('source_identity')!=receipt['source_identity'] or
     qsa_component.get('row_count')!=12 or qsa_component.get('automatic_retries')!=0 or
     qsa_component.get('after_allocated')!=0 or qsa_component.get('after_reserved')!=0):
  raise RuntimeError('Round11 QSA component gate rebind mismatch')"""
    return replace_once(source, old, new, "QSA context-reference gate rebind")


def patch_qwen(source: str) -> str:
    source = replace_once(
        source,
        "from sglang.srt.layers.round9.qsa import select_tokens, sparse_gqa, text_rope",
        "from sglang.srt.layers.round9.qsa import (select_tokens, sparse_gqa, sparse_gqa_candidate, text_rope)",
        "QSA candidate import",
    )
    source = replace_once(
        source,
        "int(self.attn.layer_id), q, keys, values, selected, positions, sparse_gqa)",
        "int(self.attn.layer_id), q, keys, values, selected, positions, sparse_gqa_candidate)",
        "QSA context consumer",
    )
    source = replace_once(
        source,
        "else sparse_gqa(q, keys, values, selected))",
        "else sparse_gqa_candidate(q, keys, values, selected))",
        "QSA direct consumer",
    )
    compile(source, "qwen4_exp.py", "exec")
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


def refresh_overlay(root: Path, qsa_source: Path) -> None:
    overlay = root / "overlay-python"
    qsa_target = overlay / "sglang/srt/layers/round9/qsa.py"
    shutil.copy2(qsa_source, qsa_target)
    qwen = overlay / "sglang/srt/models/qwen4_exp.py"
    qwen.write_text(patch_qwen(qwen.read_text()))
    tools = root / "source/tools"
    sys.path.insert(0, str(tools))
    try:
        from install_candidate_overlay import tree_digest, tree_files

        receipt_path = overlay / "round9-candidate-overlay.json"
        receipt = json.loads(receipt_path.read_text())
        receipt.setdefault("modules", {})["qsa.py"] = sha256(qsa_source)
        receipt["candidate_files"] = tree_files(overlay)
        receipt["candidate_tree_sha256"] = tree_digest(receipt["candidate_files"])
        receipt["round11_qsa_candidate"] = {
            "status": "INSTALLED_COMPONENT_V144_REQUEST_VALIDATION_PENDING",
            "candidate": "QUERY_AWARE_VECTORIZED_SPARSE_GQA",
            "component_analysis_sha256": COMPONENT_ANALYSIS_SHA256,
            "qsa_sha256": sha256(qsa_target),
            "qwen4_exp_sha256": sha256(qwen),
        }
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    finally:
        sys.path.remove(str(tools))


def controller_source(*, version: int = VERSION) -> str:
    return f'''"""Launch the one immutable Round11 T5 QSA candidate model process."""
import argparse,hashlib,json,pathlib,subprocess,time,traceback
from m3_cleanup import wait_for_guard_exit
p=argparse.ArgumentParser();p.add_argument('--admission',required=True);a=p.parse_args()
root=pathlib.Path(__file__).resolve().parent;base=pathlib.Path('/mnt/data1/kt-round9')
out=base/'evidence'/'20260926-round11-t5-qsa-model-v{version}';name='kt-round11-t5-qsa-candidate-v{version}'
claim=root/'host-controller.json';manifest=json.loads((root/'round11-source-sha256.json').read_text())
excluded={{'round11-source-sha256.json','execution-budget.json','execution-budget.lock','host-controller.json'}}
actual={{str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest() for path in root.rglob('*') if path.is_file() and '__pycache__' not in path.parts and str(path.relative_to(root)) not in excluded and not str(path.relative_to(root)).startswith('host-controller')}}
if actual!=manifest:raise RuntimeError('Round11 T5 QSA snapshot manifest mismatch')
with claim.open('x') as stream:json.dump(dict(version={version},started=time.time(),status='LAUNCHING'),stream)
args=['python3',str(root/'source/tools/launch_hybrid.py'),'--version','{version}',
 '--storage','MXFP4','--placement','static','--resource-mode','shared',
 '--shared-training-container','{TRAINING_ID}','--bank-receipt',str(root/'rebound/bank-receipt.json'),
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
 supervisor=['python3',str(root/'round11_t5_qsa_model_supervisor.py'),'--version','{version}',
  '--container',start['name'],'--container-id',start['container_id'],'--out',str(out),
  '--driver-root','/round9/'+str(root.relative_to(base))]
 subprocess.run(supervisor,check=True,timeout=21780)
 report['status']=json.loads((out/'round11-t5-qsa-model-supervisor.json').read_text())['status']
except BaseException as error:
 report.update(status='FAIL',error=repr(error),traceback=traceback.format_exc())
 try:report['failed_launch_cleanup']=cleanup_failed_launch()
 except BaseException as cleanup:report['cleanup_error']=repr(cleanup)
 raise
finally:report['ended']=time.time();save()
'''


def supervisor_source(*, version: int = VERSION) -> str:
    source = '''"""Supervise immutable Round11 T5 QSA candidate v__VERSION__."""
import argparse,json,pathlib,subprocess,time,traceback
from m3_cleanup import wait_for_guard_exit
p=argparse.ArgumentParser();p.add_argument('--version',type=int,required=True);p.add_argument('--container',required=True);p.add_argument('--container-id',required=True);p.add_argument('--out',type=pathlib.Path,required=True);p.add_argument('--driver-root',required=True);a=p.parse_args()
if (a.version,a.container)!=(__VERSION__,'kt-round11-t5-qsa-candidate-v__VERSION__') or len(a.container_id)!=64:raise ValueError('fixed Round11 T5 QSA identity required')
def command(args,timeout=30):return subprocess.check_output(args,text=True,timeout=timeout)
def inspect():
 document=json.loads(command(['docker','inspect',a.container]))[0]
 if document['Id']!=a.container_id:raise RuntimeError('owned container identity changed')
 return document
report={'status':'WAITING_FOR_SERVER','version':__VERSION__,'container_id':a.container_id,'started_at':time.time(),'generation_driver_starts':0}
path=a.out/'round11-t5-qsa-model-supervisor.json';path.touch(exist_ok=False)
def save():path.write_text(json.dumps(report,indent=2)+'\\n')
save();child=None;raw=None
try:
 deadline=time.time()+21600
 while time.time()<deadline:
  document=inspect()
  if not document['State']['Running']:raise RuntimeError('server exited before QSA model completion')
  if any(file.stat().st_size>512*1024**2 for file in a.out.glob('*.log')):raise RuntimeError('log ceiling exceeded')
  log=a.out/'server.log'
  if child is None and log.exists():
   with log.open('rb') as stream:stream.seek(max(0,log.stat().st_size-65536));tail=stream.read().decode(errors='replace')
   if 'Uvicorn running on http://127.0.0.1:30091' in tail or 'The server is fired up and ready to roll!' in tail:
    if (a.out/'round11-t5-qsa-model-raw.json').exists():raise RuntimeError('QSA model ledger already exists; no retry')
    args=['docker','exec',a.container_id,'python',a.driver_root+'/round11_t5_qsa_model_driver.py','--out','/out','--fixture',a.driver_root+'/round10-t1-fixture.json','--parent-identity','ROUND11_HFINAL_V107_ORIGINAL_C128_MXFP4','--version','__VERSION__']
    report.update(status='RUNNING_QSA_MODEL',generation_driver_starts=1,driver_command=args);save()
    with (a.out/'round11-t5-qsa-model-driver.log').open('x') as stream:child=subprocess.Popen(args,stdout=stream,stderr=subprocess.STDOUT)
  if child is not None and child.poll() is not None:
   report['driver_exit']=child.returncode;raw_path=a.out/'round11-t5-qsa-model-raw.json'
   if not raw_path.exists():raise RuntimeError('QSA model driver exited before publishing raw result')
   raw=json.loads(raw_path.read_text())
   if raw.get('generation_attempts')!=4 or raw.get('automatic_retries')!=0 or raw.get('profiler')!='OFF':raise RuntimeError('QSA model result budget mismatch')
   report['status']='DRIVER_PASS_WAITING_CLEANUP' if child.returncode==0 and raw.get('status')=='PASS_ROUND11_T5_QSA_MODEL_DRIVER_RAW' else 'FAIL'
   break
  time.sleep(5)
 else:raise TimeoutError('bounded QSA model supervisor deadline')
except BaseException as error:report.update(status='FAIL',error=repr(error),traceback=traceback.format_exc())
finally:
 report['ended_at']=time.time();save()
 try:
  document=inspect();stop_started=time.time();was_running=document['State']['Running']
  if was_running:subprocess.run(['docker','stop','-t','10',a.container_id],check=True,timeout=30)
  report['owned_exit_state']=inspect()['State'];report['guard_exit_ack']=wait_for_guard_exit(a.out/'monitor.jsonl',not_before=stop_started if was_running else report['started_at'])
  subprocess.run(['docker','rm',a.container_id],check=True,timeout=30);report['owned_container_removed']=True
  proc=command(['npu-smi','info','-t','proc-mem','-i','6','-c','0']);(a.out/'post-cleanup-proc-mem.txt').write_text(proc)
  if 'No process in device' not in proc:raise RuntimeError('physical12 is not empty after QSA model cleanup')
  report['physical12_after']='EMPTY'
 except BaseException as cleanup:report['cleanup_error']=repr(cleanup);report['status']='FAIL_CLEANUP'
 if child is not None and child.poll() is None:
  try:child.wait(timeout=20)
  except subprocess.TimeoutExpired:child.terminate()
 if report['status']=='DRIVER_PASS_WAITING_CLEANUP' and raw is not None:
  final=dict(raw);final['status']='PASS_ROUND11_T5_QSA_MODEL_PROCESS';final['cleanup']={'guard_exit_ack':True,'container_removed':True,'physical12_after':'EMPTY'}
  (a.out/'round11-t5-qsa-model-final.json').write_text(json.dumps(final,indent=2)+'\\n');report['status']='PASS_ROUND11_T5_QSA_MODEL_PROCESS'
 save()
if report['status']!='PASS_ROUND11_T5_QSA_MODEL_PROCESS':raise SystemExit(1)
'''
    source = source.replace("__VERSION__", str(version))
    compile(source, "round11_t5_qsa_model_supervisor.py", "exec")
    return source


def build(parent: Path, output: Path, *, version: int = VERSION) -> Path:
    if version not in SUPPORTED_VERSIONS:
        raise ValueError(f"QSA model snapshot must use one of {SUPPORTED_VERSIONS}")
    if output.exists():
        raise FileExistsError(output)
    local = Path(__file__).resolve().parents[2]
    component = local / "artifacts/round11/ROUND11_T5_QSA_COMPONENT_V144_ANALYSIS.json"
    if sha256(component) != COMPONENT_ANALYSIS_SHA256:
        raise ValueError("QSA v144 component evidence identity drifted")
    shutil.copytree(parent, output, ignore=shutil.ignore_patterns("__pycache__"))
    for path in (output, *output.rglob("*")):
        path.chmod(path.stat().st_mode | stat.S_IWUSR)
    for name in ("controller-F.json", "execution-budget.lock", "host-controller.json"):
        path = output / name
        if path.exists():
            path.unlink()
    qualified_qsa = output / "round11-qualified-qsa.py"
    shutil.copy2(parent / "source/runtime/qsa.py", qualified_qsa)
    launcher = output / "source/tools/launch_hybrid.py"
    launcher_source = patch_launcher(launcher.read_text(), version=version)
    if version == 146:
        launcher_source = patch_context_gate(launcher_source)
    launcher.write_text(launcher_source)
    shutil.copy2(local / "artifacts/round11/round11_monitor_owned.py", output / "source/tools/monitor_owned_server.py")
    qsa_source = local / "migration_bridge/round9/runtime/qsa.py"
    shutil.copy2(qsa_source, output / "source/runtime/qsa.py")
    refresh_overlay(output, qsa_source)
    if version == 146:
        analysis_source = local / "artifacts/round11/ROUND11_T5_QSA_COMPONENT_V144_ANALYSIS.json"
        result_source = local / "artifacts/round11/t5-qsa-results/v144/result.json"
        if sha256(result_source) != COMPONENT_RESULT_SHA256:
            raise ValueError("QSA v144 component result identity drifted")
        shutil.copy2(analysis_source, output / "ROUND11_T5_QSA_COMPONENT_V144_ANALYSIS.json")
        shutil.copy2(result_source, output / "ROUND11_T5_QSA_COMPONENT_V144_RESULT.json")
        proof = {
            "schema": 1,
            "status": "PASS_ROUND11_QSA_GATE_REBIND",
            "qualified_qsa_sha256": sha256(qualified_qsa),
            "candidate_qsa_sha256": sha256(qsa_source),
            "component_analysis_sha256": COMPONENT_ANALYSIS_SHA256,
            "component_result_sha256": COMPONENT_RESULT_SHA256,
            "semantics": (
                "ORIGINAL_CONTEXT_GATE_FOR_ALL_BASE_SOURCES_PLUS_ARCHIVED_QSA; "
                "V144_COMPONENT_GATE_FOR_CANDIDATE_QSA"
            ),
        }
        (output / "ROUND11_QSA_GATE_REBIND.json").write_text(
            json.dumps(proof, indent=2) + "\n"
        )
    shutil.copy2(local / "artifacts/round11/round11_t4_driver.py", output / "round11_t4_driver.py")
    shutil.copy2(local / "artifacts/round11/round11_t5_qsa_model_driver.py", output / "round11_t5_qsa_model_driver.py")
    (output / "host_controller.py").write_text(controller_source(version=version))
    (output / "round11_t5_qsa_model_supervisor.py").write_text(supervisor_source(version=version))
    fixture = json.loads((output / "round10-t1-fixture.json").read_text())
    if len(fixture["cases"]["edge128"]["input_ids"]) != 128:
        raise ValueError("HFINAL parent is missing the exact edge128 fixture")
    framework = json.loads((output / "framework-profile.json").read_text())
    if (
        framework.get("resident_capacity") != 32
        or framework.get("prefill", {}).get("threshold_tokens") != 129
        or framework.get("prefill", {}).get("update_residents") is not False
        or framework.get("ple", {}).get("mode") != "ASYNC"
    ):
        raise ValueError("HFINAL framework identity drifted")
    budget = {
        "schema": 1,
        "stage": "ROUND11_T5_QSA_MODEL",
        "version": version,
        "model_starts_limit": 1,
        "model_starts_used": 0,
        "starts": [],
        "generation_limit": 4,
        "automatic_retries_limit": 0,
    }
    (output / "execution-budget.json").write_text(json.dumps(budget, indent=2) + "\n")
    contract = {
        "schema": 1,
        "status": "FROZEN_NOT_RUN",
        "stage": "ROUND11_T5_QSA_REQUEST_VALIDATION",
        "version": version,
        "parent_identity": PARENT_IDENTITY,
        "candidate": "QUERY_AWARE_VECTORIZED_SPARSE_GQA",
        "component_analysis_sha256": COMPONENT_ANALYSIS_SHA256,
        "changed_runtime_files": [
            "source/runtime/qsa.py",
            "overlay-python/sglang/srt/layers/round9/qsa.py",
            "overlay-python/sglang/srt/models/qwen4_exp.py",
        ],
        "profiler": "OFF",
        "generation_limit": 4,
        "automatic_retries": 0,
        "unchanged": [
            "model", "tokenizer", "MXFP4 bank", "Static Q32 map",
            "Hybrid threshold129", "chunk32", "PLE ASYNC", "Dynamic OFF",
            "C128 NUMA2/3/4/7", "QSA indexer", "KV pool", "selected token IDs",
            "precision", "Decode sparse GQA row implementation",
        ],
        "promotion_scope": "REQUEST_INTEGRITY_AND_T6_ENTRY_ONLY",
        "repair_lineage": (
            {
                "v145": "PRELAUNCH_OLD_CONTEXT_GATE_REJECTED_CHANGED_QSA_ZERO_MODEL_STARTS",
                "v146": "SPLIT_BASE_CONTEXT_GATE_FROM_V144_CANDIDATE_QSA_GATE",
            }
            if version == 146 else None
        ),
    }
    (output / "ROUND11_T5_QSA_MODEL_CONTRACT.json").write_text(json.dumps(contract, indent=2) + "\n")
    manifest = source_manifest(output)
    (output / "round11-source-sha256.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version", type=int, default=VERSION)
    args = parser.parse_args()
    root = build(args.parent, args.output, version=args.version)
    print("PASS_ROUND11_T5_QSA_MODEL_SNAPSHOT", root, sha256(root / "round11-source-sha256.json"), flush=True)


if __name__ == "__main__":
    main()
