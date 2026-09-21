"""Bounded four-shard/two-connection-per-shard screen, no publication."""
import hashlib
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from range_recovery import ROOT, CHUNK, fetch_batch, fetch_range, append_range
from download_guard import REV, download_environment


def run_shards(fetch, rows, shard_workers):
    if shard_workers not in (1, 4):
        raise ValueError('bounded one/four shard comparison only')
    with ThreadPoolExecutor(max_workers=shard_workers) as pool:
        return list(pool.map(fetch, rows))


def main():
    stage = ROOT/'multishard-abba-v1'
    stage.mkdir(exist_ok=False)
    manifest = json.loads((ROOT/'download_source.json').read_text())
    assert manifest['revision'] == REV
    names = [f'model-{i:05d}-of-00131.safetensors' for i in (48, 49, 50, 51)]
    rows = [next(r for r in manifest['files'] if r['path'] == name) for name in names]
    protocol = dict(shard_workers=[1, 4, 4, 1], connections_per_shard=2,
                    bytes_per_shard=2*CHUNK, bytes_per_cell=8*CHUNK,
                    revision=REV, rows=rows,
                    selection='both gains >1, geometric gain >=1.10, no retries, identical prefixes')
    (stage/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    env = download_environment(os.environ)
    results = []
    for index, workers in enumerate(protocol['shard_workers']):
        cell = stage/str(index)
        cell.mkdir()
        def download(row):
            target = cell/row['path']
            started = time.monotonic()
            ranges = fetch_batch(lambda off: fetch_range(row, off, env, cell), [0, CHUNK], 2)
            for offset, end, part, header in ranges:
                append_range(target, part, header.read_text(), offset, end, row['size'])
            return dict(path=row['path'], seconds=time.monotonic()-started,
                        sha256=hashlib.sha256(target.read_bytes()).hexdigest())
        started = time.monotonic()
        files = run_shards(download, rows, workers)
        elapsed = time.monotonic()-started
        result = dict(cell=index, shards=workers, connections=2*workers,
                      seconds=elapsed, MBps=8*CHUNK/elapsed/1e6, files=files)
        results.append(result)
        (stage/'results.json').write_text(json.dumps(results, indent=2)+'\n')
        print(json.dumps(result), flush=True)
    for name in names:
        hashes = {f['sha256'] for r in results for f in r['files'] if f['path']==name}
        if len(hashes) != 1:
            raise ValueError('prefix identity mismatch: '+name)
    gains = [results[1]['MBps']/results[0]['MBps'], results[2]['MBps']/results[3]['MBps']]
    gain = math.sqrt(gains[0]*gains[1])
    errors = len(list(stage.glob('*/*.failure.json')))
    decision = dict(results=results, paired_gains=gains, geometric_gain=gain,
                    retry_errors=errors, selected_shard_workers=4 if min(gains)>1 and gain>=1.10 and not errors else 1,
                    scope='shared-network bounded screen; not sustained eight-connection qualification')
    (stage/'decision.json').write_text(json.dumps(decision, indent=2)+'\n')
    print(json.dumps(decision), flush=True)


if __name__ == '__main__':
    main()
