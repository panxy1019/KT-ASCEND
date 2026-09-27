"""Bounded transport recovery; published model files require official identity."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from curl_shard_pilot import validate_range
from download_guard import ROOT, REV, MODEL, guarded, download_environment, write_state
from verify_download import verify_file

STAGE = ROOT/'range-recovery-v1'
CHUNK = 32 * 2**20


def fetch_batch(fetch, starts, workers):
    if workers not in (1, 2):
        raise ValueError('only one or two connections are qualified candidates')
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fetch, starts))


def fetch_range(row, start, env, stage=STAGE):
    endpoint = env.get('HF_ENDPOINT', 'https://huggingface.co')
    if endpoint not in ('https://huggingface.co', 'https://hf-mirror.com'):
        raise ValueError('unapproved range endpoint')
    end = min(start+CHUNK, row['size'])-1
    for attempt in range(3):
        stem = f"{row['path']}.{start}.{time.time_ns()}"
        part, headers = stage/(stem+'.part'), stage/(stem+'.headers')
        url = f"{endpoint}/Qwen/Qwen3.8-Flash-Next/resolve/{REV}/{row['path']}"
        cmd = ['/usr/bin/curl', '--disable', '--http1.1', '--fail', '--location',
               '--silent', '--show-error', '--proto', '=https', '--proto-redir', '=https',
               '--header', 'Accept-Encoding: identity', '--range', f'{start}-{end}',
               # Shared-link transfers can progress below 32MiB/90s. Abort
               # sustained stalls, not ordinary slow ranges; retain a hard cap.
               '--connect-timeout', '15', '--max-time', '600',
               '--speed-limit', '1024', '--speed-time', '60',
               # A tiny tail can be smaller than the resolver's redirect body.
               # Keep transport bounded; validate the final body exactly below.
               '--max-filesize', str(CHUNK), '--dump-header', str(headers),
               '--output', str(part), url]
        try:
            guarded(cmd, f"RANGE:{row['path']}:{start}:{attempt}", env)
        except RuntimeError as exc:
            (stage/(stem+'.failure.json')).write_text(json.dumps(dict(error=str(exc)))+'\n')
            if attempt == 2:
                raise
            time.sleep(2*(attempt+1))
        else:
            validate_range(headers.read_text(), start, end, row['size'], part.stat().st_size)
            return start, end, part, headers


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


def shard_wave(fetch, rows, workers):
    if workers not in (1, 2, 4, 8) or len({r['path'] for r in rows}) != len(rows):
        raise ValueError('invalid shard concurrency or duplicate ownership')
    def attempt(row):
        try:
            fetch(row)
        except RuntimeError:
            return row
        return None
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return [row for row in pool.map(attempt, rows) if row is not None]


def download_row(row, env, workers):
    if (MODEL/row['path']).exists():
        if (MODEL/row['path']).stat().st_size != row['size']:
            raise ValueError('existing published file size mismatch')
        return  # full final audit verifies all existing completed files
    target = STAGE/row['path']
    if not target.exists():
        seeds = list((MODEL/'.cache/huggingface/download').glob('*.'+row['sha256']+'.incomplete'))
        if len(seeds) > 1:
            raise ValueError('ambiguous seed')
        if seeds:
            if seeds[0].stat().st_size > row['size']:
                raise ValueError('oversized seed')
            with seeds[0].open('rb') as source, target.open('xb') as out:
                shutil.copyfileobj(source, out, 2**20)
                out.flush(); os.fsync(out.fileno())
    start = target.stat().st_size if target.exists() else 0
    if start > row['size']:
        raise ValueError('oversized stage')
    while start < row['size']:
        starts = list(range(start, min(start+workers*CHUNK, row['size']), CHUNK))
        results = fetch_batch(lambda offset: fetch_range(row, offset, env), starts, workers)
        for offset, end, part, headers in results:
            append_range(target, part, headers.read_text(), offset, end, row['size'])
            part.unlink()
            start = end+1
            progress = STAGE/(row['path']+'.progress.json')
            tmp = progress.with_suffix('.tmp')
            tmp.write_text(json.dumps(dict(file=row['path'], bytes=start,
                total=row['size'], workers=workers, updated=time.time()))+'\n')
            tmp.replace(progress)
    # Identity failures must propagate as hard failures, not transport retries.
    receipt = publish(STAGE, MODEL, row)
    (STAGE/(row['path']+'.verified.json')).write_text(json.dumps(receipt)+'\n')
    print(json.dumps(receipt), flush=True)


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
    env = download_environment(os.environ)
    workers = int(os.environ.get('KT_RANGE_WORKERS', '1'))
    if workers not in (1, 2):
        raise ValueError('invalid connection count')
    # HF can fetch metadata after weights: fetch missing metadata from the same
    # frozen allowlist, verify its Git blob/SHA identity, never overwrite existing.
    for row in rows:
        if not row['path'].endswith('.safetensors'):
            if not (MODEL/row['path']).exists():
                if row['size'] > CHUNK:
                    raise ValueError('metadata requires explicit size review')
                meta = STAGE/('metadata-'+str(time.time_ns()))
                meta.mkdir()
                url = f"{env['HF_ENDPOINT']}/{manifest['repo']}/resolve/{REV}/{row['path']}"
                guarded(['/usr/bin/curl', '--disable', '--http1.1', '--fail', '--location',
                         '--silent', '--show-error', '--proto', '=https', '--proto-redir', '=https',
                         '--header', 'Accept-Encoding: identity', '--max-time', '90',
                         '--max-filesize', str(row['size']), '--output', str(meta/row['path']), url],
                        'METADATA:'+row['path'], env)
                receipt = publish(meta, MODEL, row)
                (meta/'receipt.json').write_text(json.dumps(receipt)+'\n')
            verify_file(MODEL, row)
    shard_workers = int(os.environ.get('KT_SHARD_WORKERS', '1'))
    if shard_workers not in (1, 2, 4, 8):
        raise ValueError('invalid shard count')
    pending = [row for row in rows if row['path'].endswith('.safetensors')]
    for concurrency in [n for n in (8, 4, 2, 1) if n <= shard_workers]:
        write_state(status='SHARD_WAVE', shards=concurrency, connections_per_shard=workers)
        pending = shard_wave(lambda row: download_row(row, env, workers), pending, concurrency)
        (STAGE/'wave-summary.json').write_text(json.dumps(dict(concurrency=concurrency,
            failed=[r['path'] for r in pending], updated=time.time()))+'\n')
        if not pending:
            break
        time.sleep(10)
    if pending:
        raise RuntimeError('transport retry budget exhausted: '+str([r['path'] for r in pending]))
    guarded([sys.executable, str(ROOT/'verify_download.py'), str(manifest_path),
             str(MODEL), str(ROOT/'evidence')], 'IDENTITY_VERIFY', env)
    write_state(status='DOWNLOAD_IDENTITY_PASS', model_execution='NOT_RUN')


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        write_state(status='RANGE_RECOVERY_FAILED', error=str(exc))
        raise
