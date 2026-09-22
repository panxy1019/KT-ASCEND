"""Frozen QD6 only: reference then same-history Hybrid, never starts held-out."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    root = Path('/workspace/results/r65-r66-qd6-v1')
    frozen = Path('/workspace/results/r65-r66-study-freeze-v1')
    expected = {
        'qd6.json': 'b70449cc6f744307458c21d4463c8496c5ba09e60aa21e686831b3217cf29570',
        'CONTRACT_V1.json': '4a374b76690ccbafc4fb26006463eb416e80b06a3b834992595e67365393c79b',
    }
    for name, digest in expected.items():
        if hashlib.sha256((frozen/name).read_bytes()).hexdigest() != digest:
            raise ValueError('FROZEN_INPUT_DRIFT:' + name)
    root.mkdir(exist_ok=False)
    report = dict(status='RUNNING',events=[],heldout_started=False,quality_started=False,
                  frozen_sha256=expected)
    def save():
        (root/'progress.json').write_text(json.dumps(report,indent=2)+'\n')
    try:
        save()
        for mode, profile in [('npu','four-R64'),('hybrid','four-R32')]:
            args = [sys.executable,
                    '/workspace/r65-tools/migration_bridge/round6_5/tools/capture_static_logits.py',
                    '--inputs',str(frozen/'qd6.json'),'--profile',
                    '/workspace/results/r65-profiles-v1/'+profile+'.json',
                    '--output',str(root/mode)]
            if mode == 'hybrid':
                args += ['--force',str(root/'npu/force.json')]
            report['status']='RUNNING_'+mode.upper();save()
            # No test-native extension is permitted in model qualification.
            env=dict(os.environ,KT_R65_STAGEWISE='1')
            if 'r66-debug' in env.get('PYTHONPATH',''):
                raise ValueError('DIAGNOSTIC_EXTENSION_IN_SERVING')
            with (root/(mode+'.log')).open('x') as log:
                subprocess.run(args,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            result=json.loads((root/mode/'capture-result.json').read_text())
            lifecycle=json.loads((root/mode/'lifecycle.json').read_text())
            if (result['status']!='CAPTURE_INTEGRITY_PASS' or len(result['rows'])!=12
                    or result['forward_count']!=780 or lifecycle['status']!='COMPLETE'
                    or not lifecycle['reaping']['complete']):
                raise ValueError('CAPTURE_OR_LIFECYCLE_FAILURE')
            report['events'].append(mode+'_CAPTURE_INTEGRITY_PASS');save()
        report['status']='QD6_CAPTURED_AWAIT_SYSTEM_AND_NUMERICAL_AUDIT'
    except BaseException as exc:
        report.update(status='BLOCKED',error=repr(exc));raise
    finally:
        save()


if __name__=='__main__':
    main()
