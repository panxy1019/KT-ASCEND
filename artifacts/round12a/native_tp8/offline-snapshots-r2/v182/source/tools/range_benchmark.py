"""Fixed-budget ABBA transport screen; never publishes model weights."""
import hashlib
import json
import math
import os
import time
from range_recovery import ROOT, CHUNK, fetch_batch, fetch_range, append_range
from download_guard import download_environment


def main():
    stage = ROOT / 'range-throughput-abba-v1'
    stage.mkdir(exist_ok=False)
    manifest = json.loads((ROOT / 'download_source.json').read_text())
    row = next(r for r in manifest['files'] if r['path'] == 'model-00048-of-00131.safetensors')
    protocol = dict(workers=[1, 2, 2, 1], bytes_per_cell=4*CHUNK,
                    selection='both paired gains > 1 and geometric mean gain >= 1.10',
                    source=row, revision=manifest['revision'])
    (stage / 'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    env = download_environment(os.environ)
    results = []
    for index, workers in enumerate(protocol['workers']):
        cell = stage / str(index)
        cell.mkdir()
        target = cell / 'assembled'
        started = time.monotonic()
        for start in range(0, 4*CHUNK, workers*CHUNK):
            ranges = fetch_batch(lambda off: fetch_range(row, off, env, cell),
                                 list(range(start, start+workers*CHUNK, CHUNK)), workers)
            for offset, end, part, header in ranges:
                append_range(target, part, header.read_text(), offset, end, row['size'])
        elapsed = time.monotonic()-started
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        results.append(dict(cell=index, workers=workers, seconds=elapsed,
                            MBps=4*CHUNK/elapsed/1e6, sha256=digest))
        (stage / 'results.json').write_text(json.dumps(results, indent=2)+'\n')
        print(json.dumps(results[-1]), flush=True)
    if len({r['sha256'] for r in results}) != 1:
        raise ValueError('benchmark byte identity mismatch')
    gains = [results[1]['MBps']/results[0]['MBps'], results[2]['MBps']/results[3]['MBps']]
    gain = math.sqrt(gains[0]*gains[1])
    errors = list(stage.glob('*/*.failure.json'))
    decision = dict(results=results, paired_gains=gains, geometric_gain=gain,
                    recovered_transport_errors=len(errors),
                    selected_workers=2 if min(gains)>1 and gain>=1.10 and not errors else 1,
                    scope='bounded shared-network screening, not dedicated network benchmark')
    (stage / 'decision.json').write_text(json.dumps(decision, indent=2)+'\n')
    print(json.dumps(decision), flush=True)


if __name__ == '__main__':
    main()
