"""Bounded offline followup, including failed screens. Never starts a model."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    base = Path('/workspace/results')
    root = base/'r65-r66-final-screen-v1'
    tools = Path(__file__).parent
    output = base/'r65-r66-final-postprocess-v1'
    output.mkdir(exist_ok=False)
    started = time.monotonic()
    report = dict(status='WAITING_FOR_OWNED_SEQUENCE', final_qualification=False)
    save = lambda: (output/'progress.json').write_text(json.dumps(report,indent=2)+'\n')
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    save()
    while True:
        path = root/'progress.json'
        state = json.loads(path.read_text())['status'] if path.exists() else 'STARTING'
        guard = base/'r65-r66-final-screen-guard-v1/reaping.json'
        if state in ('BLOCKED','FINAL_SCREEN_COMPLETE_AWAIT_QUALITY_LIMITATIONS_AND_ADR_REVIEW') and guard.exists():
            break
        if time.monotonic()-started>24000:
            raise TimeoutError('FINAL_SEQUENCE_WAIT_TIMEOUT')
        time.sleep(15)
    report.update(status='COLLECTING',sequence_state=state);save()
    subprocess.run([sys.executable,str(tools/'summarize_final_screen.py'),
                    '--root',str(root),'--output',str(output/'screen_summary.json')],check=True)
    paths=[root/('quality_'+mode)/'quality-report.json' for mode in ('npu','hybrid')]
    if all(p.exists() and json.loads(p.read_text())['status']=='QUALITY_SCREEN_COLLECTION_COMPLETE' for p in paths):
        subprocess.run([sys.executable,str(tools/'audit_quality_pairs.py'),
                        '--root',str(root),'--inputs',str(base/'r65-r66-study-freeze-v1/quality.json'),
                        '--parser-dir',str(tools),'--output',str(output/'all_quality_pairs.json')],check=True)
    else:
        report['paired_audit']='NOT_RUN_INCOMPLETE_QUALITY_COLLECTION'
    runtime = base/'r65-r66-runtime-source-final-b0.json'
    drift=[]
    for item in json.loads(runtime.read_text())['files']:
        if sha(Path(item['path']))!=item['sha256']:
            drift.append(item['path'])
    approval=json.loads((base/'r65-r66-quality-execution-approval-v1.json').read_text())
    for path,digest in approval['asset_sha256'].items():
        if sha(Path(path))!=digest:drift.append(path)
    for name,digest in approval['execution_tool_sha256'].items():
        if sha(tools/name)!=digest:drift.append(str(tools/name))
    statsha=sha(tools/'paired_quality_interval.py')
    if statsha!='74eed4fc7ddf2f7831bc76a7c739ec22e6c849b4f54c52f71fd46b18c025db64':
        drift.append(str(tools/'paired_quality_interval.py'))
    report.update(status='COLLECTED_AWAIT_ADR' if not drift else 'PROVENANCE_DRIFT',
                  drift=drift,statistics_sha256=statsha,
                  artifact_sha256={p.name:sha(p) for p in output.glob('*.json') if p.name!='progress.json'})
    save()
    assert not drift,drift


if __name__=='__main__':main()
