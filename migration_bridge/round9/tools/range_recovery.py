"""Bounded transport recovery; published model files require official identity."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import time
from curl_shard_pilot import validate_range
from download_guard import ROOT, REV, MODEL, guarded, download_environment, write_state
from verify_download import verify_file

STAGE = ROOT/'range-recovery-v1'
CHUNK = 32 * 2**20


def append_range(target, part, headers, start, end, total):
    validate_range(headers, start, end, total, part.stat().st_size)
    if (target.stat().st_size if target.exists() else 0) != start:
        raise ValueError('local range offset changed')
    with target.open('ab') as out, part.open('rb') as source:
        shutil.copyfileobj(source, out, 2**20)
        out.flush()
        os.fsync(out.fileno())


def publish(stage, model, row):
    receipt = verify_file(stage, row)
    # Atomic no-clobber publication on the same filesystem, never overwrite.
    os.link(stage/row['path'], model/row['path'])
    return receipt


def main():
    manifest_path = ROOT/'download_source.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest['revision'] != REV or manifest['repo'] != 'Qwen/Qwen3.8-Flash-Next':
        raise ValueError('wrong frozen source')
    rows = manifest['files']
    if any(Path(r['path']).name != r['path'] for r in rows):
        raise ValueError('unexpected path')
    if sys.argv[1:2] == ['--publish']:
        row = next(r for r in rows if r['path'] == sys.argv[2])
        receipt = publish(STAGE, MODEL, row)
        (STAGE/(row['path']+'.verified.json')).write_text(json.dumps(receipt)+'\n')
        print(json.dumps(receipt), flush=True)
        return
    if sys.argv[1:]:
        raise ValueError('unknown arguments')
    STAGE.mkdir(exist_ok=True)
    lock = (STAGE/'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # Existing metadata must remain intact; never silently switch discovery source.
    for row in rows:
        if not row['path'].endswith('.safetensors'):
            verify_file(MODEL, row)
    env = download_environment(os.environ)
    for row in rows:
        if not row['path'].endswith('.safetensors'):
            continue
        if (MODEL/row['path']).exists():
            if (MODEL/row['path']).stat().st_size != row['size']:
                raise ValueError('existing published file size mismatch')
            continue  # full final audit verifies all existing completed files
        target = STAGE/row['path']
        if not target.exists():
            seeds = list((MODEL/'.cache/huggingface/download').glob('*.'+row['sha256']+'.incomplete'))
            if len(seeds) > 1:
                raise ValueError('ambiguous seed')
            if seeds:
                if seeds[0].stat().st_size > row['size']:
                    raise ValueError('oversized seed')
                # Copy, not rename: original failure evidence/cache remains untouched.
                with seeds[0].open('rb') as source, target.open('xb') as out:
                    shutil.copyfileobj(source, out, 2**20)
                    out.flush(); os.fsync(out.fileno())
        start = target.stat().st_size if target.exists() else 0
        if start > row['size']:
            raise ValueError('oversized stage')
        while start < row['size']:
            end = min(start+CHUNK, row['size'])-1
            for attempt in range(3):
                stem = f"{row['path']}.{start}.{time.time_ns()}"
                part, headers = STAGE/(stem+'.part'), STAGE/(stem+'.headers')
                url = f"https://huggingface.co/{manifest['repo']}/resolve/{REV}/{row['path']}"
                cmd = ['/usr/bin/curl', '--disable', '--http1.1', '--fail', '--location',
                       '--silent', '--show-error', '--proto', '=https', '--proto-redir', '=https',
                       '--header', 'Accept-Encoding: identity', '--range', f'{start}-{end}',
                       '--connect-timeout', '15', '--max-time', '90',
                       '--max-filesize', str(end-start+1), '--dump-header', str(headers),
                       '--output', str(part), url]
                try:
                    guarded(cmd, f"RANGE:{row['path']}:{start}:{attempt}", env)
                    append_range(target, part, headers.read_text(), start, end, row['size'])
                except RuntimeError as exc:
                    # Only subprocess/transport failure retries; invalid ranges hard-stop.
                    (STAGE/(stem+'.failure.json')).write_text(json.dumps(dict(error=str(exc)))+'\n')
                    if attempt == 2:
                        raise
                    time.sleep(2*(attempt+1))
                else:
                    part.unlink()  # fully represented in fsynced stage, headers retained
                    break
            start = end+1
            (STAGE/'progress.json').write_text(json.dumps(dict(file=row['path'], bytes=start,
                                                            total=row['size'], updated=time.time()))+'\n')
        guarded([sys.executable, str(Path(__file__).resolve()), '--publish', row['path']],
                'SHARD_IDENTITY:'+row['path'], env)
    guarded([sys.executable, str(ROOT/'verify_download.py'), str(manifest_path),
             str(MODEL), str(ROOT/'evidence')], 'IDENTITY_VERIFY', env)
    write_state(status='DOWNLOAD_IDENTITY_PASS', model_execution='NOT_RUN')


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        write_state(status='RANGE_RECOVERY_FAILED', error=str(exc))
        raise
