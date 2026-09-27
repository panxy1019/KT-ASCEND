"""One missing shard via TLS-verified bounded ranges; no model publication."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
from download_guard import ROOT, REV, MODEL, guarded, write_state
from verify_download import verify_file


def validate_range(headers, start, end, total, size):
    blocks = re.split(r'(?m)^HTTP/', headers)
    final = blocks[-1] if len(blocks) > 1 else ''
    status = re.match(r'\S+\s+(\d+)', final)
    field = re.search(r'(?im)^content-range:\s*bytes (\d+)-(\d+)/(\d+)\s*$', final)
    if (not status or status.group(1) != '206' or not field or
            tuple(map(int, field.groups())) != (start, end, total) or
            size != end - start + 1):
        raise ValueError('RANGE_IDENTITY_OR_LENGTH_MISMATCH')
    return True


def main():
    out = ROOT / 'curl-shard-pilot-v1'
    if sys.argv[1:] == ['--verify']:
        receipt = verify_file(out, json.loads((out/'selected.json').read_text()))
        (out/'verified.json').write_text(json.dumps(receipt, indent=2)+'\n')
        return
    assert not sys.argv[1:]
    manifest_path = ROOT/'download_source.json'
    manifest = json.loads(manifest_path.read_text())
    assert manifest['revision'] == REV and manifest['repo'] == 'Qwen/Qwen3.8-Flash-Next'
    row = next(x for x in manifest['files'] if x['path'].endswith('.safetensors')
               and not (MODEL/x['path']).exists())
    assert Path(row['path']).name == row['path'] and re.fullmatch(r'[a-f0-9]{64}', row['sha256'])
    assert 0 < row['size'] < 8 * 2**30
    out.mkdir(exist_ok=False)
    (out/'selected.json').write_text(json.dumps(row, indent=2)+'\n')
    (out/'provenance.json').write_text(json.dumps(dict(revision=REV,
        manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        start=time.time(), model_publication=False), indent=2)+'\n')
    target = out/row['path']
    chunk_bytes = 64 * 2**20
    url = f"https://hf-mirror.com/{manifest['repo']}/resolve/{REV}/{row['path']}"
    with target.open('xb') as assembled:
        for start in range(0, row['size'], chunk_bytes):
            end = min(start + chunk_bytes, row['size']) - 1
            part, header = out/f'{start}.part', out/f'{start}.headers'
            command = ['/usr/bin/curl', '--disable', '--fail', '--location',
                '--silent', '--show-error', '--proto', '=https', '--proto-redir', '=https',
                '--range', f'{start}-{end}', '--connect-timeout', '15', '--max-time', '180',
                '--limit-rate', '20M', '--max-filesize', str(end-start+1),
                '--dump-header', str(header), '--output', str(part), url]
            # No blanket retry: retain the exact first transport/range failure.
            guarded(command, f'CURL_PILOT_RANGE_{start}', dict(os.environ))
            validate_range(header.read_text(), start, end, row['size'], part.stat().st_size)
            with part.open('rb') as source:
                shutil.copyfileobj(source, assembled, 2**20)
            assembled.flush()
            os.fsync(assembled.fileno())
            # Exact owned chunk is now represented in assembled; keep headers.
            part.unlink()
            (out/'progress.json').write_text(json.dumps(dict(bytes=end+1, total=row['size']))+'\n')
    guarded([sys.executable, str(Path(__file__).resolve()), '--verify'],
            'CURL_PILOT_IDENTITY', dict(os.environ))
    write_state(status='SINGLE_SHARD_PILOT_VERIFIED', model_execution='NOT_RUN')


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        write_state(status='CURL_PILOT_FAILED', error=str(exc))
        raise
