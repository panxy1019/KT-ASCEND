"""Read-only host telemetry, scoped to the project container and NPU0.

Writes only a new project evidence file. Device counters are shared-device
aggregate, not exclusive project consumption. Never signals any process.
"""
import json
from pathlib import Path
import subprocess
import time


def main():
    root = Path('/home/admin/kt-artifacts/round6-deferred')
    output = root / 'r63-pilot-retry1-resource-samples.jsonl'
    final = root / 'r63-pilot-retry1-pipeline/result.json'
    observer = root / 'r63-observer-correction-retry1-pipeline/result.json'
    deadline = time.monotonic() + 3600
    def read(command):
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                universal_newlines=True, timeout=20)
        return dict(returncode=result.returncode, output=result.stdout)
    with output.open('x') as stream:
        while True:
            item = dict(unix=time.time(), scope='project container process RSS; NPU0 aggregate',
                processes=read(['docker', 'top', 'kt-r6-deferred-ascend', '-eo', 'pid,pgid,rss,time,comm']),
                npu0=read(['npu-smi', 'info', '-t', 'usages', '-i', '0']))
            if final.exists():
                item['terminal'] = json.loads(final.read_text())['status']
            elif observer.exists() and json.loads(observer.read_text())['status'] != 'OBSERVER_CORRECTION_PASS':
                item['terminal'] = 'OBSERVER_FAILED_NO_PILOT'
            elif time.monotonic() >= deadline:
                item['terminal'] = 'SAMPLER_TIME_BUDGET'
            stream.write(json.dumps(item) + '\n')
            stream.flush()
            if 'terminal' in item:
                break
            time.sleep(10)


if __name__ == '__main__':
    main()
