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
from common import sha, write


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
    paths = ['/proc/meminfo', '/proc/vmstat', '/sys/fs/cgroup/memory.current', '/sys/fs/cgroup/memory.peak',
             '/sys/fs/cgroup/memory.stat', '/sys/fs/cgroup/memory.events', '/sys/fs/cgroup/memory.pressure']
    processes=[]
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            command=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
            if not any(x in command for x in ('sglang','scheduler')):continue
            status=(proc/'status').read_text()
            masks={}
            for task in (proc/'task').iterdir():
                for line in (task/'status').read_text().splitlines():
                    if line.startswith('Cpus_allowed_list:'):
                        mask=line.split(':',1)[1].strip();masks[mask]=masks.get(mask,0)+1
            locality={'anon':{},'file':{}}
            for line in (proc/'numa_maps').read_text().splitlines():
                kind='file' if 'file=' in line else 'anon'
                for field in line.split():
                    if field.startswith('N') and '=' in field and field[1:field.index('=')].isdigit():
                        node,count=field.split('=');locality[kind][node]=locality[kind].get(node,0)+int(count)
            processes.append(dict(pid=int(proc.name),command=command,status=status,
                numa_pages=locality,thread_affinity_counts=masks,stat=(proc/'stat').read_text()))
        except (FileNotFoundError,ProcessLookupError,PermissionError):pass
    return dict(time_unix=time.time(),files={p:Path(p).read_text() for p in paths if Path(p).exists()},
                processes=processes)


def info():
    # This deployed SGLang's idle get_internal_state RPC did not return within
    # 30s on the raw R64 server. It is optional telemetry, not an inference gate.
    # No runtime-memory measurement is fabricated from launch configuration.
    return dict(status='UNAVAILABLE', reason='IDLE_SERVER_INFO_RPC_TIMEOUT_IN_V3',
                device_memory_source='HOST_NPU_SMI_AND_OBSERVER_FRAMEWORK_COUNTERS')


def validate_launch(root, observer):
    resolved = json.loads((root / 'resolved-command.json').read_text())
    args = resolved['argv']
    required = {'--max-total-tokens':'4096', '--max-running-requests':'1',
                '--tp-size':'1', '--dtype':'bfloat16', '--device':'npu',
                '--kt-cpuinfer':str(int(os.environ['C64_TP'])*32),
                '--kt-threadpool-count':os.environ['C64_TP'],
                '--kt-max-deferred-experts-per-token':'0'}
    for key,value in required.items():
        if key not in args or args[args.index(key)+1] != value:
            raise ValueError('LAUNCH_CONFIGURATION_DRIFT:' + key)
    if '--disable-cuda-graph' not in args or ('--forward-hooks' in args) != observer:
        raise ValueError('GRAPH_OR_OBSERVER_DRIFT')
    expected=['4','5'] if os.environ['C64_TP']=='2' else ['4']
    index=args.index('--kt-numa-nodes')+1
    if args[index:index+len(expected)]!=expected:raise ValueError('NUMA_LAUNCH_DRIFT')
    return dict(configured_kv_token_capacity=4096, graph_off=True, batch_limit=1,
                evidence='RESOLVED_LAUNCH_ARGS_NOT_DYNAMIC_MEMORY', resolved_sha256=sha(root / 'resolved-command.json'))


def check_port_available(port=31000):
    # Match the retained launcher and HTTP server: TIME_WAIT is reusable, a
    # live listening socket is not. Never terminate another port owner.
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(('127.0.0.1', port))


@contextmanager
def session(root, profile, observer=False):
    root = Path(root); root.mkdir(parents=True, exist_ok=False)
    admitted = admission()
    check_port_available()
    tools = Path(__file__).parent
    cmd = [sys.executable, str(tools/'owned_guard.py'),
           '--output', str(root / 'guard'), '--timeout-seconds', '43200', '--',
           'taskset', '-c', os.environ.get('KT_R67_CPU_SET', '0-31'), 'bash', str(tools / 'launch_serial.sh'),
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
            report.update(status='READY', launch_configuration=validate_launch(root, observer),
                          memory_after_load=memory(), server_after_load=info())
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
                    report['preceding_status'] = report['status']
                    report['status'] = 'OWNED_LIFECYCLE_FAILURE'
            report.update(ended_unix=time.time(), memory_after_stop=memory())
            write(root / 'lifecycle.json', report)
            if report['status'] == 'OWNED_LIFECYCLE_FAILURE': raise RuntimeError('OWNED_LIFECYCLE_FAILURE')
