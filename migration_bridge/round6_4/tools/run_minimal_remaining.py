"""Sequential project-only continuation; stop at any hard failure, never refit."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT=Path('/workspace/results/r64-minimal-completion-v2')
TOOLS=Path(__file__).parent
FIXTURES=TOOLS.parent/'fixtures'
PY='/workspace/results/venv-r6/bin/python'


def main():
    ROOT.mkdir(exist_ok=False)
    report=dict(status='WAITING_QUALITY',completed=[],started_unix=time.time())
    def save():
        (ROOT/'progress.json').write_text(json.dumps(report,indent=2)+'\n')
    def call(name,args,capacity=True):
        report.update(status='RUNNING',active=name)
        save()
        command=(["bash",str(TOOLS/'run_capacity_test.sh')] if capacity else [PY,str(TOOLS/args[0])])
        command += args if capacity else args[1:]
        with (ROOT/(name+'.log')).open('x') as log:
            subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
        report['completed'].append(name)
        save()
    try:
        save()
        summary=Path('/workspace/results/r64-fast-quality-summary-v1.json')
        deadline=time.monotonic()+10800
        while not summary.exists():
            for mode in ('off','on'):
                path=Path(f'/workspace/results/r64-fast-quality-{mode}-v1/quality-report.json')
                if path.exists() and json.loads(path.read_text())['status']=='FAILED':
                    raise RuntimeError('QUALITY_EXECUTION_FAILURE:'+mode)
            if time.monotonic()>deadline:
                raise TimeoutError('QUALITY_SUMMARY_NOT_AVAILABLE')
            time.sleep(10)
        quality=json.loads(summary.read_text())
        if quality['status']!='QUALITY_SCREEN_PASS':
            raise RuntimeError('QUALITY_SCREEN_NOT_PASS:'+quality['status'])
        report['quality_summary_sha256']=hashlib.sha256(summary.read_bytes()).hexdigest()
        parity='/workspace/results/r64-minimal-perf-parity-v1'
        call('performance_parity',['run_performance_observer_parity.py','--corpus','/workspace/results/corpora-v2/s6.json',
            '--performance-manifest',str(FIXTURES/'performance-fast.json'),'--output',parity])
        pr=json.loads((Path(parity)/'parity-report.json').read_text())
        assert pr['status']=='PERFORMANCE_OBSERVER_PARITY_PASS' and pr['same_workload_deferred_route_hits']>0
        entry=ROOT/'performance-entry.json'
        entry.write_text(json.dumps(dict(status='R64_PERFORMANCE_ENTRY_APPROVED',
            manifest_raw_sha256=hashlib.sha256((FIXTURES/'performance-fast.json').read_bytes()).hexdigest(),
            parity_report_sha256=hashlib.sha256((Path(parity)/'parity-report.json').read_bytes()).hexdigest(),
            quality_summary_sha256=report['quality_summary_sha256']),indent=2)+'\n')
        call('performance',['run_performance_campaign.py','--manifest',str(FIXTURES/'performance-fast.json'),
            '--entry-audit',str(entry),'--output','/workspace/results/r64-minimal-performance-v1'])
        for campaign in ('a','b'):
            command=['run_stability_campaign.py','--corpus','/workspace/results/corpora-v2/s6.json',
                '--output',f'/workspace/results/r64-minimal-stability-{campaign}-v1']
            if campaign=='b':
                command+=['--reference','/workspace/results/r64-minimal-stability-a-v1/stability-report.json']
            call('stability_'+campaign,command)
        call('descriptive',['run_descriptive_holdout.py','--inputs','/workspace/results/round6_3-formal-inputs',
            '--entry-audit',str(FIXTURES/'descriptive_entry.json'),'--output','/workspace/results/r64-descriptive-final-v1'])
        call('descriptive_analysis',['analyze_descriptive_holdout.py','--root','/workspace/results/r64-descriptive-final-v1'],False)
        report['status']='EXECUTION_COMPLETE_PENDING_RESOURCE_REVIEW_AND_ADR'
    except BaseException as exc:
        report.update(status='STOPPED_FIRST_FAILURE',exception=repr(exc))
        raise
    finally:
        report['ended_unix']=time.time()
        save()


if __name__=='__main__':
    main()
