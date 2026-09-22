"""Single cold Dynamic cell, attribution only; never formal timing evidence."""
from pathlib import Path
import json
import sys
from qwen_dynamic_retry_release import frozen_source, replace_once, audit
from run_qwen_dynamic_perf_retry import adapted_controller, retry_config

ROOT = Path('/home/admin/kt-round8-qwen35')
NAME = 'qwen-dynamic-resource-diagnostic-v1'


def cgroup_path(text):
    lines = [line[3:] for line in text.splitlines() if line.startswith('0::')]
    if len(lines) != 1 or not lines[0].startswith('/') or lines[0] == '/' or '..' in Path(lines[0]).parts:
        raise ValueError('unsafe or missing owned unified cgroup')
    return Path('/sys/fs/cgroup') / lines[0].lstrip('/')


def snapshot():
    from run_qwen_abba import snapshot as base, command
    row = base()
    info = json.loads(command('docker', 'inspect', 'kt-round8-qwen35-prep'))[0]
    assert info['State']['Running']
    pid = info['State']['Pid']
    group = cgroup_path(Path(f'/proc/{pid}/cgroup').read_text())
    row['owned_cgroup_path'] = str(group)
    row['owned_cgroup'] = {name: (group/name).read_text() for name in
        ('memory.current', 'memory.stat', 'memory.events', 'memory.swap.current',
         'memory.pressure', 'memory.numa_stat', 'cpu.stat', 'io.stat')}
    row['numa'] = {str(n): {name: Path(f'/sys/devices/system/node/node{n}/{name}').read_text()
                          for name in ('meminfo', 'vmstat')} for n in (4, 5)}
    row['owned_processes'] = {}
    for value in (group/'cgroup.procs').read_text().split():
        try:
            row['owned_processes'][value] = {name: Path(f'/proc/{value}/{name}').read_text()
                                           for name in ('status', 'stat', 'io')}
        except FileNotFoundError:
            row['owned_processes'][value] = {'exited_during_sample': True}
    row['formal_performance'] = False
    return row


def controller(source):
    source = adapted_controller(source)
    for old, new in (
        ("NAME='qwen-dynamic-perf-v3'", f"NAME='{NAME}'"),
        ('for block in range(3):', 'for block in range(1):'),
        ("enumerate(('static','dynamic','dynamic','static'))", "enumerate(('dynamic',))"),
        ('from run_qwen_abba import snapshot, ensure_stopped', 'from run_qwen_abba import ensure_stopped'),
        ("'/workspace/round8/tools/run_qwen_dynamic_perf_retry.py','--client',",
         "'/workspace/round8/tools/qwen_resource_diagnostic.py','--client',"),
        ("for line in row['containers'].splitlines():",
         "assert int(row['owned_cgroup']['memory.swap.current'])==0,'OWNED_SWAP'\n"
         "                        assert all(int(v)==0 for k,v in (s.split() for s in row['owned_cgroup']['memory.events'].splitlines()) if k in ('oom','oom_kill')),'OWNED_OOM'\n"
         "                        for line in row['containers'].splitlines():"),
        ("state['completed'].append(cell);save();ensure_stopped();owns_process=False",
         "with (out/f'{cell}-resources.jsonl').open('a') as end: end.write(json.dumps(snapshot())+'\\n')\n"
         "                state['completed'].append(cell);save();ensure_stopped();owns_process=False"),
        ("state['status']='COLLECTED_PENDING_RESOURCE_AUDIT'",
         "state['status']='DIAGNOSTIC_COLLECTED_PENDING_AUDIT'"),
    ):
        source = replace_once(source, old, new)
    return source


def main():
    if '--client' in sys.argv:
        sys.argv.remove('--client')
        source = replace_once(frozen_source(Path('/workspace/round8'), 'qwen_dynamic_perf.py'),
                              "'qwen-dynamic-perf-v2'", repr(NAME))
    else:
        assert json.loads((ROOT/'qwen-dynamic-perf-v3/state.json').read_text())['status'] == 'COLLECTED_PENDING_RESOURCE_AUDIT'
        assert Path('/home/admin/kt-round9-qwen38/PAUSE_DOWNLOAD').exists()
        assert all(audit(ROOT)[mode]['status'] == 'PASS' for mode in ('heldout', 'cmmlu'))
        source = controller(frozen_source(ROOT, 'run_qwen_dynamic_perf.py'))
    exec(compile(source, 'frozen_controller[single-cell-resource-diagnostic]', 'exec'),
         {'__name__': '__main__', 'snapshot': snapshot, 'retry_config': retry_config})


if __name__ == '__main__': main()
