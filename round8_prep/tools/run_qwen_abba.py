"""Host controller: twelve distinct ABBA cells, owned container only, fail closed."""
import fcntl
import json
import subprocess
import time
from pathlib import Path

ROOT = Path('/home/admin/kt-round8-qwen35')
CONTAINER = 'kt-round8-qwen35-prep'
CAMPAIGN = 'qwen-static-perf-v3'
PYTHON = '/workspace/round8/venv/bin/python'
TOOLS = '/workspace/round8/tools'


def command(*args):
    return subprocess.check_output(args, text=True, timeout=120)


def cells():
    return [(b, p, profile) for b in range(3)
            for p, profile in enumerate(('q64', 'q32', 'q32', 'q64'))]


def snapshot():
    return dict(timestamp=time.time(),
        npu=command('npu-smi', 'info'),
        containers=command('docker', 'stats', '--no-stream', '--format', '{{json .}}'),
        meminfo=Path('/proc/meminfo').read_text(),
        vmstat=Path('/proc/vmstat').read_text(),
        pressure=Path('/proc/pressure/memory').read_text())


def ensure_stopped():
    stopped = subprocess.run(['docker', 'stop', '-t', '60', CONTAINER],
                             text=True, capture_output=True, timeout=120)
    inspect = json.loads(command('docker', 'inspect', CONTAINER))[0]
    if inspect['State']['Running']:
        raise RuntimeError(f'owned container still running after stop: {stopped.stderr}')
    # Docker may report "did not receive an exit event" even though inspect
    # already records a cleanly reaped, non-OOM stopped container.
    return dict(returncode=stopped.returncode, stdout=stopped.stdout,
                stderr=stopped.stderr, state=inspect['State'])


def main():
    out = ROOT / CAMPAIGN
    out.mkdir(exist_ok=True)
    lock = (out / 'controller.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (out / 'controller-state.json').exists(), 'Use new identity after failure; do not overwrite'
    state = dict(status='RUNNING', completed=[], started=time.time())
    def save():
        (out / 'controller-state.json').write_text(json.dumps(state, indent=2)+'\n')
    save()
    try:
        for block, position, profile in cells():
            name = f'{profile}-block{block}-position{position}'
            state['current'] = name
            save()
            # Every cell starts a clean serving process, preserving all mounted artifacts.
            stop = ensure_stopped()
            (out/f'{name}-stop.json').write_text(json.dumps(stop, indent=2)+'\n')
            command('docker', 'start', CONTAINER)
            subprocess.run(['python3', str(ROOT/'tools/admit_q64_host.py')], check=True,
                           stdout=(out/f'{name}-admission.log').open('w'), timeout=180)
            command('docker', 'exec', '-d', CONTAINER, 'bash', '-c',
                f'exec {PYTHON} {TOOLS}/start_q64.py --candidate {profile}-perf-launch-v2 '
                f'--admission /workspace/round8/q64-admission-v1.json '
                f'> /workspace/round8/qwen-static-perf-v2/{name}-server.log 2>&1')
            deadline = time.monotonic()+1200
            while True:
                ready = subprocess.run(['docker','exec',CONTAINER,PYTHON,'-c',
                    'import requests; assert requests.get("http://127.0.0.1:18400/health",timeout=3).status_code==200'],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)
                if ready.returncode == 0:
                    break
                assert time.monotonic()<deadline, 'Server readiness timeout'
                time.sleep(5)
            with (out/f'{name}-client.log').open('w') as log:
                worker = subprocess.Popen(['docker','exec',CONTAINER,PYTHON,
                    f'{TOOLS}/qwen_static_perf.py','--profile',profile,'--block',str(block),
                    '--position',str(position),'--campaign',CAMPAIGN],
                    stdout=log,stderr=subprocess.STDOUT)
                with (out/f'{name}-resources.jsonl').open('w') as resource:
                    while worker.poll() is None:
                        resource.write(json.dumps(snapshot())+'\n')
                        resource.flush()
                        time.sleep(10)
                assert worker.wait()==0, 'Performance client failed'
            summary=json.loads((out/name/'summary.json').read_text())
            assert summary['status']=='BLOCK_COMPLETE' and len(summary['rows'])==3
            state['completed'].append(name)
            save()
        state['status']='COLLECTION_COMPLETE_PENDING_RESOURCE_AUDIT'
    except BaseException as error:
        state['status']='FAILED'
        state['error']=repr(error)
        raise
    finally:
        state['updated']=time.time()
        save()


if __name__=='__main__':
    main()
