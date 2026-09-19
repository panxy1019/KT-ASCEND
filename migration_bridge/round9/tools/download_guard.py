"""Own only a new download/verification process group; yield to Round8 prep lane."""
import json
import os
import pathlib
import shutil
import signal
import subprocess
import sys
import time

ROOT = pathlib.Path('/home/admin/kt-round9-qwen38')
REV = 'de4b8e4d43b917e7706784d8bb445c9af86a3540'
MODEL = ROOT / 'models' / REV
HF = '/home/admin/.local/bin/hf'


def write_state(**data):
    data.update(updated=time.time(), revision=REV, controller_pid=os.getpid())
    tmp = ROOT / 'download-state.tmp'
    tmp.write_text(json.dumps(data) + '\n')
    tmp.replace(ROOT / 'download-state.json')


def pause_reason():
    if (ROOT / 'PAUSE_DOWNLOAD').exists():
        return 'EXPLICIT_PERFORMANCE_RESERVATION'
    query = subprocess.run(['docker', 'inspect', 'kt-round8-qwen35-prep',
                            '--format', '{{.State.Running}}'],
                           capture_output=True, text=True, timeout=15)
    if query.returncode or query.stdout.strip() not in ('true', 'false'):
        return 'UNKNOWN_ROUND8_STATE'
    if query.stdout.strip() == 'true':
        return 'ROUND8_PREP_LANE_ACTIVE'
    available = next(int(line.split()[1]) * 1024
                     for line in pathlib.Path('/proc/meminfo').read_text().splitlines()
                     if line.startswith('MemAvailable:'))
    if available < 128 * 2**30:
        return 'HOST_MEMORY_RESERVE'
    if shutil.disk_usage(ROOT).free < 150 * 2**30:
        return 'DISK_RESERVE'
    return None


def guarded(command, phase, env):
    while True:
        reason = pause_reason()
        if reason is None:
            break
        write_state(phase=phase, status='PAUSED', reason=reason)
        time.sleep(5)
    child = subprocess.Popen(command, env=env, start_new_session=True)
    paused = False
    try:
        while child.poll() is None:
            reason = pause_reason()
            if bool(reason) != paused:
                os.killpg(child.pid, signal.SIGSTOP if reason else signal.SIGCONT)
                paused = bool(reason)
            write_state(phase=phase, status='PAUSED' if paused else 'RUNNING',
                        reason=reason, child_pgid=child.pid)
            time.sleep(5)
        if child.returncode:
            raise RuntimeError(f'{phase} exited {child.returncode}')
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGCONT)
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()


def download_command(manifest):
    return [HF, 'download', manifest['repo'], *[x['path'] for x in manifest['files']],
            '--revision', REV, '--local-dir', str(MODEL), '--max-workers', '1']


def download_environment(source):
    endpoint = source.get('KT_DOWNLOAD_ENDPOINT', 'https://hf-mirror.com')
    if endpoint not in ('https://hf-mirror.com', 'https://huggingface.co'):
        raise ValueError('unapproved download endpoint')
    return dict(source, HF_ENDPOINT=endpoint,
               HF_HOME=str(ROOT / 'hf-home'), HF_HUB_DISABLE_XET='1',
               HF_HUB_DISABLE_TELEMETRY='1', HF_HUB_DISABLE_IMPLICIT_TOKEN='1',
               HF_HUB_DOWNLOAD_TIMEOUT='120', HF_HUB_ETAG_TIMEOUT='60',
               HF_HUB_DISABLE_PROGRESS_BARS='1')


def main():
    MODEL.mkdir(parents=True, exist_ok=True)
    env = download_environment(os.environ)
    manifest = ROOT / 'download_source.json'
    m = json.loads(manifest.read_text())
    if m['revision'] != REV or m['repo'] != 'Qwen/Qwen3.8-Flash-Next':
        raise ValueError('wrong frozen source')
    # Explicit filenames keep mirror repository additions outside this experiment.
    guarded(download_command(m), 'DOWNLOAD', env)
    guarded([sys.executable, str(ROOT / 'verify_download.py'), str(manifest),
             str(MODEL), str(ROOT / 'evidence')], 'IDENTITY_VERIFY', env)
    write_state(status='DOWNLOAD_IDENTITY_PASS', model_execution='NOT_RUN')


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        write_state(status='FAILED', error=str(exc))
        raise
