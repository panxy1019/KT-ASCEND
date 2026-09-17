"""Owned long-enough lifecycle, no unscoped process termination."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from common import write


def admission():
    fields = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':', 1)
        if value.strip().endswith('kB'): fields[key] = int(value.split()[0]) * 1024
    if fields['SwapTotal'] - fields['SwapFree'] != 0:
        raise ValueError('SWAP_USED')
    if fields['MemAvailable'] < max(64 * 2**30, fields['MemTotal'] * .2):
        raise ValueError('HOST_MEMORY_RESERVE')
    return fields


def memory():
    paths = ['/proc/meminfo', '/proc/vmstat', '/sys/fs/cgroup/memory.current',
             '/sys/fs/cgroup/memory.stat', '/sys/fs/cgroup/memory.events', '/sys/fs/cgroup/memory.pressure']
    return dict(time_unix=time.time(), files={p:Path(p).read_text() for p in paths if Path(p).exists()})


def info():
    with urllib.request.urlopen('http://127.0.0.1:31000/server_info', timeout=30) as r: data = json.load(r)
    states = data['internal_states']
    if len(states) != 1 or states[0]['memory_usage']['token_capacity'] != 4096 or states[0]['memory_usage']['graph'] != 0:
        raise ValueError('KV_OR_GRAPH_DRIFT')
    return data


@contextmanager
def session(root, profile, observer=False):
    root = Path(root); root.mkdir(parents=True, exist_ok=False)
    admitted = admission()
    with socket.socket() as s: s.bind(('127.0.0.1', 31000))
    tools = Path(__file__).parent
    cmd = [sys.executable, '/workspace/r65-tools/migration_bridge/round6_5/tools/owned_guard.py',
           '--output', str(root / 'guard'), '--timeout-seconds', '43200', '--',
           'taskset', '-c', '0-31', 'bash', str(tools / 'launch_serial.sh'),
           str(root), str(profile), str(int(observer))]
    report = dict(status='STARTING', command=cmd, started_unix=time.time(), admission=admitted,
                  memory_before_load=memory())
    child = None
    with (root / 'launcher.log').open('x') as log:
        try:
            child = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + 1200
            while True:
                if child.poll() is not None: raise RuntimeError('MODEL_START_FAILED')
                if time.monotonic() > deadline: raise TimeoutError('MODEL_READY_TIMEOUT')
                try:
                    with urllib.request.urlopen('http://127.0.0.1:31000/health', timeout=2) as r:
                        if r.status == 200: break
                except OSError: pass
                time.sleep(2)
            report.update(status='READY', memory_after_load=memory(), server_after_load=info())
            write(root / 'ready.json', report)
            yield root
            report.update(status='COMPLETE', memory_after_work=memory(), server_after_work=info())
        except BaseException as exc:
            report.update(status='FAILED', error=repr(exc)); raise
        finally:
            if child is not None:
                if child.poll() is None: child.send_signal(signal.SIGTERM)
                child.wait(timeout=60)
                report['reaping'] = json.loads((root / 'guard/reaping.json').read_text())
                report['guard'] = json.loads((root / 'guard/result.json').read_text())
                if not report['reaping']['complete'] or report['guard'].get('reason') not in ('CANCELLED', 'EXIT_ZERO'):
                    report['status'] = 'OWNED_LIFECYCLE_FAILURE'
            report.update(ended_unix=time.time(), memory_after_stop=memory())
            write(root / 'lifecycle.json', report)
            if report['status'] == 'OWNED_LIFECYCLE_FAILURE': raise RuntimeError('OWNED_LIFECYCLE_FAILURE')
