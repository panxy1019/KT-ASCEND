"""Audit complete raw ABBA cells before applying Round7's declared default gate.

Pure Python, no device work. Missing/contended windows are pending evidence, not
proof of no benefit. A coalesced SSE stream is not a decode measurement.
"""
import argparse
import json
import math
from pathlib import Path
import statistics


BUCKETS = ('short', 'medium', 'long')
BUDGETS = dict(short=256, medium=512, long=1024)
ORDER = ('left', 'right', 'right', 'left')


def positive(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def evaluate(report, isolation):
    if report.get('status') != 'COMPLETE' or report.get('observer') is not False or report.get('serial') is not True:
        raise ValueError('INCOMPLETE_OR_OBSERVED_PERFORMANCE')
    rows = report['measurements']
    expected = {(b, block, turn) for b in BUCKETS for block in range(3) for turn in range(4)}
    keys = [(r['bucket'], r['block'], r['turn']) for r in rows]
    if len(keys) != 36 or len(set(keys)) != 36 or set(keys) != expected:
        raise ValueError('ABBA_CELL_COMPLETENESS')
    if keys != [(b, block, turn) for b in BUCKETS for block in range(3) for turn in range(4)]:
        raise ValueError('ABBA_CELL_ORDER')
    modes = {}
    for row in rows:
        if row['side'] != ORDER[row['turn']]:
            raise ValueError('ABBA_ORDER')
        modes.setdefault(row['side'], set()).add(row['mode'])
        if not positive(row['request_seconds']):
            raise ValueError('INVALID_LATENCY')
        if len(row['output_ids']) != BUDGETS[row['bucket']]:
            raise ValueError('DECODE_BUDGET')
        rate = row['decode_tokens_per_second']
        if rate is not None and not positive(rate):
            raise ValueError('INVALID_DECODE_RATE')
    if any(len(v) != 1 for v in modes.values()):
        raise ValueError('MIXED_MODE')
    for bucket in BUCKETS:
        same = [r for r in rows if r['bucket'] == bucket]
        if len({(r['prompt_id'], r['input_tokens']) for r in same}) != 1:
            raise ValueError('WORKLOAD_IDENTITY')

    windows = isolation.get('windows', [])
    window_keys = [(w['bucket'], w['block'], w['turn']) for w in windows]
    eligible = (len(window_keys) == 36 and len(set(window_keys)) == 36 and
                set(window_keys) == expected and all(w.get('formal_eligible') is True for w in windows))
    workloads = {}
    for bucket in BUCKETS:
        sides = {side: [r for r in rows if r['bucket'] == bucket and r['side'] == side]
                 for side in ('left', 'right')}
        latency = {side: statistics.median(r['request_seconds'] for r in group)
                   for side, group in sides.items()}
        # Do not estimate a median from a non-random subset of uncoalesced runs.
        decode = {side: statistics.median(r['decode_tokens_per_second'] for r in group)
                  if all(r['decode_tokens_per_second'] is not None for r in group) else None
                  for side, group in sides.items()}
        gain_latency = 100 * (1 - latency['right'] / latency['left'])
        gain_decode = (100 * (decode['right'] / decode['left'] - 1)
                       if all(v is not None for v in decode.values()) else None)
        blocks = []
        for block in range(3):
            times = {side: statistics.median(r['request_seconds'] for r in group if r['block'] == block)
                     for side, group in sides.items()}
            blocks.append(dict(block=block, latency_improvement_percent=100 * (1 - times['right'] / times['left'])))
        # User's recommended dynamic default threshold; no threshold fitting.
        gain = gain_latency >= 3 or (gain_decode is not None and gain_decode >= 3)
        # Conservatively reject median regressions on either available axis,
        # rather than asserting an unmeasured regression is absent.
        regression = latency['right'] / latency['left'] > 1.05 or (
            gain_decode is not None and decode['right'] / decode['left'] < .95)
        workloads[bucket] = dict(median_e2e_seconds=latency, median_decode_tokens_per_second=decode,
            latency_improvement_percent=gain_latency, throughput_improvement_percent=gain_decode,
            meets_3_percent_gain=gain, median_regression_over_5_percent=regression,
            abba_blocks=blocks, positive_latency_blocks=sum(b['latency_improvement_percent'] > 0 for b in blocks))
    improved = sum(w['meets_3_percent_gain'] for w in workloads.values())
    regressed = any(w['median_regression_over_5_percent'] for w in workloads.values())
    return dict(status='PERFORMANCE_EVIDENCE_AUDITED' if eligible else 'PENDING_HOST_ISOLATION',
        modes={k: next(iter(v)) for k, v in modes.items()}, workloads=workloads,
        host_isolation_complete=eligible, improved_workloads=improved,
        median_regression_over_5_percent=regressed,
        recommended_dynamic_default_gate=('PASS' if improved >= 2 and not regressed else 'NO_BENEFIT') if eligible else 'PENDING',
        scope='PERFORMANCE_ONLY_NOT_SYSTEM_QUALITY_OR_FINAL_PRODUCT_DECISION',
        limitations=['Client SSE TTFT is not exact internal prefill time.',
                     'Host isolation is a contention proxy, not direct DRAM bandwidth.',
                     'Block-level gains are reported for repeatability review; quality and migration identity are independent gates.'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--summary', type=Path, required=True)
    p.add_argument('--isolation', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    result = evaluate(json.loads(a.summary.read_text()), json.loads(a.isolation.read_text()))
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(status=result['status'], recommended_dynamic_default_gate=result['recommended_dynamic_default_gate'])))


if __name__ == '__main__':
    main()
