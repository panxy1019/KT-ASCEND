"""Wait for the frozen .22 model identity, then replicate and verify it on A3."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


REV = 'de4b8e4d43b917e7706784d8bb445c9af86a3540'
SOURCE_HOST = '110.129.0.22'
SOURCE_ROOT = Path('/home/admin/kt-round9-qwen38')
DEST_ROOT = Path('/home/admin/models/Qwen3.8-Flash-Next')
STAGE = DEST_ROOT / ('.' + REV + '.partial')
FINAL = DEST_ROOT / REV
SUPPORT = DEST_ROOT / (REV + '.round9-support')
STATE = SUPPORT / 'replication-state.json'
MIN_FREE = 450 * 10**9


def atomic_state(**values):
    SUPPORT.mkdir(parents=True, exist_ok=True)
    values.update(updated=time.time(), revision=REV, source_host=SOURCE_HOST,
                  stage=str(STAGE), final=str(FINAL))
    tmp = STATE.with_suffix('.tmp')
    tmp.write_text(json.dumps(values, indent=2) + '\n')
    tmp.replace(STATE)


def ssh_env():
    password = os.environ.get('KT_SOURCE_SSHPASS')
    if not password:
        raise ValueError('KT_SOURCE_SSHPASS is required')
    return dict(os.environ, SSHPASS=password)


def ssh_command(*remote):
    return ['/usr/bin/sshpass', '-e', '/usr/bin/ssh',
            '-o', 'ConnectTimeout=15', '-o', 'ServerAliveInterval=30',
            '-o', 'ServerAliveCountMax=4', '-o', 'StrictHostKeyChecking=accept-new',
            f'admin@{SOURCE_HOST}', *remote]


def source_state(env):
    result = subprocess.run(ssh_command('cat', str(SOURCE_ROOT/'download-state.json')),
                            env=env, check=True, capture_output=True, text=True,
                            timeout=30)
    return json.loads(result.stdout)


def wait_for_source(env):
    while True:
        try:
            state = source_state(env)
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
            atomic_state(status='WAITING_SOURCE', error=str(exc))
        else:
            status = state.get('status')
            atomic_state(status='WAITING_SOURCE', source_status=status,
                         source_updated=state.get('updated'))
            if status == 'DOWNLOAD_IDENTITY_PASS' and state.get('revision') == REV:
                return
        time.sleep(60)


def rsync_source(env, source, destination):
    command = ['/usr/bin/sshpass', '-e', '/usr/bin/rsync', '--archive',
               '--partial', '--append-verify', '--human-readable',
               '--info=progress2,stats2', '--timeout=180',
               '-e', ('/usr/bin/ssh -o ConnectTimeout=15 '
                      '-o ServerAliveInterval=30 -o ServerAliveCountMax=4 '
                      '-o StrictHostKeyChecking=accept-new'),
               f'admin@{SOURCE_HOST}:{source}', str(destination)]
    while True:
        atomic_state(status='TRANSFERRING', source=source, destination=str(destination))
        result = subprocess.run(command, env=env)
        if result.returncode == 0:
            return
        atomic_state(status='TRANSFER_RETRY', returncode=result.returncode,
                     source=source, destination=str(destination))
        time.sleep(60)


def verify_and_publish(env):
    rsync_source(env, str(SOURCE_ROOT/'download_source.json'), SUPPORT/'download_source.json')
    rsync_source(env, str(SOURCE_ROOT/'verify_download.py'), SUPPORT/'verify_download.py')
    STAGE.mkdir(parents=True, exist_ok=True)
    rsync_source(env, str(SOURCE_ROOT/'models'/REV) + '/', str(STAGE) + '/')
    evidence = SUPPORT / 'destination-verification'
    evidence.mkdir(exist_ok=True)
    atomic_state(status='VERIFYING_DESTINATION')
    result = subprocess.run([sys.executable, str(SUPPORT/'verify_download.py'),
                             str(SUPPORT/'download_source.json'), str(STAGE),
                             str(evidence)])
    if result.returncode:
        atomic_state(status='DESTINATION_IDENTITY_FAILED', returncode=result.returncode)
        raise RuntimeError('destination identity verification failed')
    if FINAL.exists():
        atomic_state(status='BLOCKED_FINAL_EXISTS')
        raise FileExistsError(FINAL)
    STAGE.rename(FINAL)
    atomic_state(status='REPLICATION_IDENTITY_PASS', model_execution='NOT_RUN')


def main():
    env = ssh_env()
    if sys.argv[1:] == ['--check-only']:
        print(json.dumps(dict(source_state=source_state(env),
                              free=shutil.disk_usage('/home/admin/models').free,
                              stage=str(STAGE), final=str(FINAL))))
        return
    if sys.argv[1:]:
        raise ValueError('unknown arguments')
    if FINAL.exists():
        atomic_state(status='BLOCKED_FINAL_EXISTS')
        raise FileExistsError(FINAL)
    free = shutil.disk_usage('/home/admin/models').free
    if free < MIN_FREE:
        atomic_state(status='BLOCKED_DISK_ADMISSION', free=free, required=MIN_FREE)
        raise RuntimeError('destination disk admission failed')
    wait_for_source(env)
    verify_and_publish(env)


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        try:
            atomic_state(status='REPLICATION_FAILED', error=str(exc))
        finally:
            raise
