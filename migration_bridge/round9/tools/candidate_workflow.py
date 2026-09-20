"""Round9 development entry point. No device launch, heavy conversion or timing.

profile writes independent configuration; plan validates supplied allowed CPU
geometry without reserving it; check emits source-bound host/tensor test evidence.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from migration_bridge.round9.runtime.candidate_config import Candidate,runtime_arguments,validate_topology

HOST=['test_candidates','test_lowbit','test_candidate_bank','test_dynamic_candidate','test_candidate_workflow']
TENSOR=['test_dynamic_tensor_adapter','test_candidate_gguf']


def write_new(path,data):
    with Path(path).open('x') as f:json.dump(data,f,indent=2);f.write('\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);commands=p.add_subparsers(dest='command',required=True)
    profile=commands.add_parser('profile');profile.add_argument('--workers',type=int,choices=[64,128],required=True)
    profile.add_argument('--storage',choices=['BF16','MXFP4','Q8_0'],required=True)
    profile.add_argument('--placement',choices=['static','request_boundary'],required=True)
    profile.add_argument('--numa-nodes',required=True);profile.add_argument('--output',type=Path,required=True)
    plan=commands.add_parser('plan');plan.add_argument('--profile',type=Path,required=True);plan.add_argument('--topology',type=Path,required=True);plan.add_argument('--output',type=Path,required=True)
    check=commands.add_parser('check');check.add_argument('--stage',choices=['host','native-partition','tensor','all'],required=True);check.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.command=='profile':
        value=dict(workers=a.workers,storage=a.storage,placement=a.placement,numa_nodes=[int(n) for n in a.numa_nodes.split(',')])
        Candidate.from_dict(value);write_new(a.output,value);print('PROFILE_WRITTEN_NOT_ADMITTED');return 0
    if a.command=='plan':
        candidate=Candidate.from_dict(json.loads(a.profile.read_text()));topology=json.loads(a.topology.read_text())
        allocation=validate_topology(candidate,topology['rows'],topology['quota'])
        report=dict(status='HOST_PLAN_ONLY_DEVICE_ADMISSION_PENDING',launch_permitted=False,
            profile=asdict(candidate),profile_sha256=hashlib.sha256(a.profile.read_bytes()).hexdigest(),
            allocation=allocation,partitions=candidate.partitions(),server_args=runtime_arguments(candidate),
            native_status=candidate.native_status(),dynamic_status='BOUNDARY_ADAPTER_HOST_TESTED_SCHEDULER_INSTALLATION_PENDING' if candidate.placement=='request_boundary' else 'OFF',
            next_gate='fresh host/training/HBM/NUMA and exact source/native/device qualification')
        write_new(a.output,report);print(json.dumps(report,indent=2));return 0
    a.output.mkdir(parents=True,exist_ok=False)
    names=HOST if a.stage=='host' else TENSOR if a.stage=='tensor' else HOST+TENSOR
    names=['migration_bridge.round9.tests.'+name for name in names]
    if a.stage=='native-partition':names=['migration_bridge.round9.tests.test_candidates.CandidateTests.test_native_four_pool_partition_matches_python']
    env=dict(os.environ,TORCH_DEVICE_BACKEND_AUTOLOAD='0',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    started=time.time()
    with (a.output/'tests.log').open('x') as f:
        process=subprocess.run([sys.executable,'-m','unittest',*names,'-v'],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=180)
    files={}
    round9=ROOT/'migration_bridge/round9'
    for directory in ('runtime','tools','tests'):
        for path in sorted((round9/directory).glob('*.py')):
            files[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    header=ROOT/'kt-kernel/operators/llamafile/tp_slices.hpp'
    files[str(header.relative_to(ROOT))]=hashlib.sha256(header.read_bytes()).hexdigest()
    report=dict(status='PASS_HOST_DEVELOPMENT_TESTS' if process.returncode==0 else 'FAIL',stage=a.stage,
        started_at=started,ended_at=time.time(),python=sys.version,platform=platform.platform(),
        command=[sys.executable,'-m','unittest',*names,'-v'],returncode=process.returncode,
        device_qualification=False,performance_qualification=False,files=files)
    write_new(a.output/'summary.json',report)
    print((a.output/'tests.log').read_text());print(json.dumps({k:v for k,v in report.items() if k!='files'},indent=2))
    return 0 if process.returncode==0 else 1

if __name__=='__main__':raise SystemExit(main())
