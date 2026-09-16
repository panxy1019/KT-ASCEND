"""Offline closure audit; no threshold derivation or model execution."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    def read(path):
        return json.loads((a.root/path).read_text())
    progress = read('progress.json')
    modes = ['off_native', 'on_forced', 'on_native']
    assert progress['completed'] == modes
    analysis = read('descriptive-report.json')
    assert analysis['status'] == 'DESCRIPTIVE_ANALYSIS_COMPLETE'
    assert analysis['numerical_acceptance_gate'] is False and analysis['quality_qualified'] is False
    requests = read('native.json')['requests']
    ids = [r['rid'] for r in requests]
    assert len(ids) == len(set(ids)) == 136
    sources, captures, rows = [], {}, []
    for mode in modes:
        root = Path(mode)
        integrity = read(root/'final-integrity.json')
        assert integrity == dict(status='PASS', physical_captures=8840, indexed=8840,
                                 targets=8704, lookahead=136, unknown=0, incomplete=0,
                                 numerical_metrics_read=False)
        lifecycle = read(root/'lifecycle.json')
        assert lifecycle['status'] == 'PLANNED_STOP' and lifecycle['cpu_workers'] == 32
        assert lifecycle['guard']['reason'] == 'CANCELLED'
        assert lifecycle['reaping']['status'] == 'OWNED_DESCENDANTS_REAPED'
        assert read(root/'readiness.json')['capture_session_empty'] is True
        result = read(root/'capture/result.json')
        assert result['status'] == 'PREQUALIFICATION_CAPTURE_PASS' and result['complete_forwards'] == 8840
        assert result['forced'] == (mode == 'on_forced')
        assert [r['rid'] for r in result['rows']] == ids
        # Raw steps include the explicit lookahead; scored/generated IDs do not.
        assert all(len(r['baseline_output_ids']) == 64 and len(r['steps']) == 65 for r in result['rows'])
        assert all(r['input_ids'] == q['input_ids'] for r,q in zip(result['rows'], requests))
        index = [json.loads(line) for line in (a.root/root/'index.jsonl').read_text().splitlines()]
        assert len(index) == len({r['file'] for r in index}) == 8840
        assert [r['sequence'] for r in index] == list(range(8840))
        captures[mode] = result['rows']
        sources.append(read(root/'provenance.json')['file_sha256'])
        rows.append(dict(mode=mode, requests=136, forward_count=8840,
                         started_unix=result['started_unix'], ended_unix=result['ended_unix']))
    common = set.intersection(*(set(s) for s in sources))
    assert all(len({s[k] for s in sources}) == 1 for k in common), 'SOURCE_DRIFT'
    assert all(x['baseline_output_ids'] == y['baseline_output_ids'] for x,y in
               zip(captures['off_native'],captures['on_forced'])), 'FORCED_HISTORY_DRIFT'
    assert [r['rid'] for r in analysis['rows']] == ids
    for row in analysis['rows']:
        assert len(row['steps']) == 64
        assert all(math.isfinite(value) for metric in row['metrics'].values() for value in metric.values())
    files = [p for p in a.root.rglob('*') if p.is_file()]
    hashes = {str(p.relative_to(a.root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
    report = dict(status='FINAL_DESCRIPTIVE_METADATA_AUDIT_PASS', modes=rows,
                  requests=408, physical_forwards=26520, target_positions=26112,
                  lookahead=408, unknown=0, incomplete=0, matched_history=True,
                  common_source_files=len(common), summaries=analysis['summaries'],
                  finite_metrics=True, quality_or_speedup_gate=False,
                  raw_tensor_reaudit='original A3 audits retained; this audit checks exported metadata',
                  file_sha256=hashes)
    with a.output.open('x') as out:
        json.dump(report,out,indent=2)
        out.write('\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('file_sha256','summaries')}))


if __name__ == '__main__':
    main()
