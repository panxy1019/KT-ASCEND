"""Build the immutable one-process Round10 T6 final regression snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import build_round10_t1_snapshot as t1


MUTABLE={"execution-budget.json","execution-budget.lock","controller-F.json","round10-source-sha256.json"}


def once(source,old,new,label):
    if source.count(old)!=1:raise ValueError(f"T6 {label} precondition mismatch")
    return source.replace(old,new)


def patch_launcher(source):
    source=once(source,"if a.version != 95 or a.cost_mode != 'OFF':","if a.version != 107 or a.cost_mode != 'OFF':","version")
    source=once(source,"name='kt-round10-t1-r-v95';out=root/'evidence'/'20260926-round10-t1-v95'",
                "name='kt-round10-t6-final-v107';out=root/'evidence'/'20260926-round10-t6-v107'","identity")
    return once(source,"if budget.get('stage')!='ROUND10_T1_PROCESS_R':raise RuntimeError('wrong Round10 T1 budget')",
                "if budget.get('stage')!='ROUND10_T6_FINAL':raise RuntimeError('wrong Round10 T6 budget')","budget")


def patch_guard(source):
    return once(source,"r'|kt-round10-t1-[abr]-v[0-9]+)', name) is not None",
                "r'|kt-round10-t1-[abr]-v[0-9]+'\n        r'|kt-round10-t6-final-v[0-9]+)', name) is not None","guard")


def patch_supervisor(source):
    source=once(source,"choices=('A','B','R')","choices=('A','B','R','F')","choices")
    source=once(source,"(('A',91),('B',92),('A',93),('B',94),('R',95))",
                "(('A',91),('B',92),('A',93),('B',94),('R',95),('F',107))","identity")
    source=once(source,"expected_name=f'kt-round10-t1-{a.process.lower()}-v{a.version}'",
                "expected_name='kt-round10-t6-final-v107' if a.process=='F' else f'kt-round10-t1-{a.process.lower()}-v{a.version}'","name")
    source=source.replace("round10-t1-supervisor.json","round10-t6-supervisor.json")
    source=source.replace("round10-t1-results.json","round10-t6-results.json")
    source=source.replace("/round10_t1_driver.py","/round10_t6_driver.py")
    source=once(source,"{'A':11,'B':11,'R':19}[a.process]","{'A':11,'B':11,'R':19,'F':8}[a.process]","generation limit")
    source=once(source,"{'A':1,'B':1,'R':2}[a.process]","{'A':1,'B':1,'R':2,'F':1}[a.process]","abort limit")
    return source.replace("PASS_ROUND10_T1_PROCESS","PASS_ROUND10_T6_PROCESS")


def patch_controller(source):
    return once(source,"round10-t1-supervisor.json","round10-t6-supervisor.json","controller receipt")


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest(root):
    return {str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and str(p.relative_to(root)) not in MUTABLE
            and not str(p.relative_to(root)).startswith('controller-')}


def build(args):
    if args.output.exists():raise FileExistsError(args.output)
    shutil.copytree(args.parent,args.output,ignore=shutil.ignore_patterns('__pycache__'))
    launcher=args.output/'source/tools/launch_hybrid.py';launcher.write_text(patch_launcher(launcher.read_text()))
    guard=args.output/'source/tools/monitor_owned_server.py';guard.write_text(patch_guard(guard.read_text()))
    local=Path(__file__).resolve().parent
    shutil.copy2(local/'round10_t6_driver.py',args.output/'round10_t6_driver.py')
    values={'process':'F','version':107,'container_name':'kt-round10-t6-final-v107',
            'evidence_root':'/mnt/data1/kt-round9/evidence/20260926-round10-t6-v107'}
    (args.output/'host_controller.py').write_text(patch_controller(t1.controller_source(values,args.training_id)))
    (args.output/'round10_t1_supervisor.py').write_text(patch_supervisor(t1.supervisor_source()))
    budget={'schema':1,'stage':'ROUND10_T6_FINAL','model_starts_limit':1,'model_starts_used':0,
            'starts':[],'automatic_retries_limit':0}
    (args.output/'execution-budget.json').write_text(json.dumps(budget,indent=2)+'\n')
    contract={'schema':1,'status':'FROZEN_NOT_RUN','version':107,'parent':'H0_ROUND10_V95',
      'final_config':{'threshold_tokens':129,'dynamic':'OFF','cpu_backend':'C128_MXFP4_CURRENT',
       'npu_resident':'Q32_BF16_NZ29','chunk_tokens':32,'ple':'ASYNC_CURRENT','profiler':'OFF'},
      'generation_limit':8,'abort_controls':1,'model_starts_limit':1,'automatic_retries':0}
    (args.output/'ROUND10_T6_SNAPSHOT_CONTRACT.json').write_text(json.dumps(contract,indent=2)+'\n')
    files=manifest(args.output);(args.output/'round10-source-sha256.json').write_text(json.dumps(files,indent=2,sort_keys=True)+'\n')
    for path in (launcher,guard,args.output/'host_controller.py',args.output/'round10_t1_supervisor.py',args.output/'round10_t6_driver.py'):
        compile(path.read_text(),str(path),'exec')
    print('PASS_ROUND10_T6_SNAPSHOT',len(files),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--training-id',required=True);build(p.parse_args())


if __name__=='__main__':main()
