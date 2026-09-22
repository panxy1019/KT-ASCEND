"""Post-audit unchanged release-v6 timing against independently recorded host data.

Run on A3 before archiving raw files: old v6 did not record an end wallclock, so
the original output-file mtime bounds completion conservatively. This does not
infer end time from request duration or inject hooks into the timed model path.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re

from evaluate_performance import evaluate
from host_resource_observer import assess_window


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture_window(root, row):
    bucket, block, turn = row['bucket'], row['block'], row['turn']
    rid = f'measure-{bucket}-{block}-{turn}'
    directory = root / f'{bucket}-{block}-{turn}-{row["mode"]}'
    input_path = directory / f'{rid}-input.json'
    raw_path = directory / f'{rid}.json'
    original = json.loads(input_path.read_text())
    raw = json.loads(raw_path.read_text())
    if raw['payload'] != original['payload'] or raw['payload']['rid'] != rid:
        raise ValueError('RAW_REQUEST_IDENTITY')
    if raw['summary']['final_meta']['id'] != rid:
        raise ValueError('RAW_RESPONSE_IDENTITY')
    for name in ('prompt_id', 'input_tokens', 'request_seconds', 'output_ids', 'decode_tokens_per_second'):
        if raw['summary'][name] != row[name]:
            raise ValueError('RAW_SUMMARY_MISMATCH')
    lifecycle = json.loads((directory / 'lifecycle.json').read_text())
    if lifecycle['status'] != 'COMPLETE' or lifecycle['reaping']['complete'] is not True:
        raise ValueError('PERFORMANCE_PROCESS_NOT_REAPED')
    start = original['started_unix']
    end = raw_path.stat().st_mtime
    if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in (start, end)) or start >= end:
        raise ValueError('INVALID_RAW_WALLCLOCK')
    # Reject unexplained wallclock/duration disagreement; serialization belongs
    # to the conservative observer window, never the performance denominator.
    serialization_and_clock_slack = end - start - row['request_seconds']
    if not -.01 <= serialization_and_clock_slack <= 10:
        raise ValueError('WALLCLOCK_DURATION_DISAGREEMENT')
    if not lifecycle['started_unix'] <= start < end <= lifecycle['ended_unix']:
        raise ValueError('WALLCLOCK_OUTSIDE_LIFECYCLE')
    return dict(bucket=bucket, block=block, turn=turn, started_unix=start, ended_unix=end,
        endpoint_source='ORIGINAL_INPUT_WALLCLOCK_THROUGH_ORIGINAL_OUTPUT_FILE_MTIME',
        serialization_and_clock_slack_seconds=serialization_and_clock_slack,
        input_sha256=sha(input_path), raw_sha256=sha(raw_path),
        lifecycle_sha256=sha(directory / 'lifecycle.json'))


def audit(root, report, samples, observer_sha):
    # Reuse structural validation without pretending missing host data passes.
    evaluate(report, {})
    captures = [capture_window(root, row) for row in report['measurements']]
    if any(b['started_unix'] < a['ended_unix'] for a, b in zip(captures, captures[1:])):
        raise ValueError('NON_SERIAL_OR_REORDERED_CAPTURE_WINDOWS')
    first = min(x['started_unix'] for x in captures) - 20
    last = max(x['ended_unix'] for x in captures) + 20
    snapshot = [s for s in samples if first <= s['time_unix'] <= last]
    if not snapshot or any(s.get('source_sha256') != observer_sha for s in snapshot):
        raise ValueError('HOST_OBSERVER_SOURCE_IDENTITY')
    windows = [dict(**c, **assess_window(snapshot, c['started_unix'], c['ended_unix'])) for c in captures]
    isolation = dict(status='PASS' if all(w['formal_eligible'] for w in windows) else 'CONTAMINATED_OR_UNOBSERVED',
        windows=windows, observer_source_sha256=observer_sha,
        conservative_completion_boundary=True,
        scope='READ_ONLY_HOST_CONTENTION_PROXY_NOT_DIRECT_DRAM_BANDWIDTH')
    return isolation, snapshot


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--host-samples', type=Path, required=True)
    p.add_argument('--observer-source-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if not re.fullmatch('[0-9a-f]{64}', a.observer_source_sha256):
        raise ValueError('OBSERVER_SOURCE_HASH_REQUIRED')
    a.output.mkdir(parents=True, exist_ok=False)
    report = json.loads((a.root / 'summary.json').read_text())
    # Reader may race an append of a future sample; retain only complete lines.
    text = a.host_samples.read_text()
    complete = text[:text.rfind('\n') + 1]
    samples = [json.loads(line) for line in complete.splitlines()]
    isolation, snapshot = audit(a.root, report, samples, a.observer_source_sha256)
    for name, value in [('host-isolation.json', isolation),
                        ('performance-decision.json', evaluate(report, isolation)),
                        ('provenance.json', dict(summary_sha256=sha(a.root/'summary.json'),
                            host_sample_snapshot_scope='ONLY_BOUNDED_PERFORMANCE_CAMPAIGN',
                            ignored_incomplete_future_append=text != complete,
                            observer_source_sha256=a.observer_source_sha256))]:
        (a.output / name).write_text(json.dumps(value, indent=2) + '\n')
    (a.output / 'host-sample-snapshot.jsonl').write_text(''.join(json.dumps(s)+'\n' for s in snapshot))
    print(json.dumps(dict(status=isolation['status'], measured_windows=len(isolation['windows']))))


if __name__ == '__main__':
    main()
