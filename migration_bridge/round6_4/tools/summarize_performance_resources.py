"""Describe owned-phase resource observations; no automatic attribution claim."""
import argparse
import hashlib
import json
from pathlib import Path


def describe(samples, started, ended, guard_path):
    during = [s for s in samples if started <= s['time_unix'] <= ended]
    owned = [s for s in during if s['status'] == 'SAMPLED'
             and guard_path in s.get('owned_guard_paths', {}).values()]
    hbm = [sum(s['owned_hbm_mb_by_pid'].values()) for s in owned
           if s.get('owned_hbm_mb_by_pid')]
    def bounds(values):
        return [min(values), max(values)] if values else None
    return dict(samples=len(during), owned_samples=len(owned),
        monitor_errors=[s for s in during if s['status'] == 'MONITOR_ERROR'],
        owned_sample_time_range=bounds([s['time_unix'] for s in owned]),
        owned_rss_bytes_range=bounds([s['summed_owned_rss_bytes'] for s in owned]),
        owned_hbm_mb_range=bounds(hbm),
        other_npu0_hbm_mb_range=bounds([s['other_process_hbm_mb'] for s in owned]),
        host_load1_range=bounds([s['host_load_average'][0] for s in owned]),
        scope='Whole phase including startup/warmup/teardown, not timed-request-only',
        limitations=['HBM residency is not evidence of compute idleness',
                     'Summed RSS can double-count shared pages',
                     'Host load is not CPU-affinity-specific contention',
                     'Sparse samples do not prove absence of transient contention'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--remote-root', required=True)
    parser.add_argument('--memory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    campaign = json.loads((args.root/'campaign.json').read_text())
    assert campaign['status'] == 'PERFORMANCE_CAPTURE_COMPLETE_PENDING_ANALYSIS'
    samples = [json.loads(s) for s in args.memory.read_text().splitlines()]
    phases = {}
    for name in campaign['completed']:
        report = json.loads((args.root/name/'performance-report.json').read_text())
        assert report['status'] == 'PERFORMANCE_PHASE_COMPLETE'
        phases[name] = describe(samples, report['started_unix'], report['ended_unix'],
                                args.remote_root+'/'+name+'/guard')
    result = dict(status='RESOURCE_DESCRIPTION_COMPLETE_REVIEW_REQUIRED',
                  campaign_root=args.remote_root, phases=phases,
                  memory_sha256=hashlib.sha256(args.memory.read_bytes()).hexdigest(),
                  campaign_sha256=hashlib.sha256((args.root/'campaign.json').read_bytes()).hexdigest())
    with args.output.open('x') as output:
        json.dump(result, output, indent=2)
        output.write('\n')
    print(json.dumps(dict(status=result['status'], phases=len(phases))))


if __name__ == '__main__':
    main()
