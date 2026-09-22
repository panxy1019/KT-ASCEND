"""Summarize an already frozen numerical report; never fits a new threshold."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
from evaluate_stateful_contract import frozen


def distribution(values):
    if not values:
        return None
    assert all(math.isfinite(v) for v in values)
    ordered = sorted(values)
    def at(q):
        return ordered[max(0, math.ceil(q*len(ordered))-1)]
    return dict(n=len(values), p50=at(.5), p95=at(.95), p99=at(.99), maximum=ordered[-1])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    report = frozen(args.report)
    assert report['action'] == 'derive'
    rows = report['rows']
    groups = dict(all=rows, exposed=[r for r in rows if r['deferred_exposure'] != 'NO_EXPOSURE'],
                  no_exposure=[r for r in rows if r['deferred_exposure'] == 'NO_EXPOSURE'])
    summary = dict(status='QD3_DERIVED_NOT_HELDOUT_VERIFIED' if report['status'] == 'PASS' else 'QD3_REJECTED',
        report_sha256=report['sha256'], preregistration_sha256=report['preregistration_sha256'],
        corpus_sha256=report['corpus_sha256'], exposure_counts=dict(Counter(r['deferred_exposure'] for r in rows)),
        stable=sum(r['stable'] for r in rows), ambiguous=sum(not r['stable'] for r in rows),
        failures=len(report['failures']), first_failures=report['failures'][:10], reserves=report['reserves'],
        descriptive_distributions={group:{metric:distribution([r[metric] for r in values])
            for metric in report['reserves']} for group,values in groups.items()},
        subgroup_statistics='descriptive only; no additional unregistered confidence claim',
        runtime='NOT_ESTABLISHED_BY_THIS_REPORT', heldout='NOT_ESTABLISHED_BY_THIS_REPORT',
        quality='NOT_ESTABLISHED_BY_THIS_REPORT')
    args.output.mkdir(exist_ok=False, parents=True)
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2))
    lines = ['# QD3 numerical derivation summary', '', summary['status'], '',
             f"Report SHA: `{report['sha256']}`", '',
             '| Metric | All p50 | All p95 | All p99 | All max | Frozen reserve |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for metric, reserve in report['reserves'].items():
        value = summary['descriptive_distributions']['all'][metric]
        lines.append(f"|{metric}|{value['p50']}|{value['p95']}|{value['p99']}|{value['maximum']}|{reserve['threshold']}|")
    lines += ['', f"Stable positions: {summary['stable']}; ambiguous: {summary['ambiguous']}; hard failures: {summary['failures']}.",
              '', 'Exposed/no-exposure subgroup distributions are descriptive only.',
              'A QD3 PASS is not runtime, held-out, quality or Round6.3 approval.']
    (args.output/'SUMMARY.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
