"""Host-side verified collection of relocated system probes, not timing data."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def collect(sources, output, expected_provenance):
    reference=json.loads(expected_provenance.read_text())
    records=[]
    for threshold,root,provenance in sources:
        summary=json.loads((root/'summary.json').read_text())
        protocol=json.loads((root/'protocol.json').read_text())
        assert summary['status']=='PASS' and summary['threshold']==threshold
        assert protocol['threshold']==threshold
        actual=json.loads(provenance.read_text())
        for key in ('files','gguf','cpu_native_sha256','bf16_context_port'):
            assert actual[key]==reference[key],f'WORKER_ARTIFACT_DRIFT:{key}'
        for process in ('process-a','process-b','observer-off'):
            lifecycle=json.loads((root/process/'lifecycle.json').read_text())
            assert lifecycle['status']=='COMPLETE' and lifecycle['reaping']['complete']
        records.append(dict(threshold=threshold,source=str(root),
            protocol=protocol,summary_sha256=digest(root/'summary.json'),
            provenance_sha256=digest(provenance),
            artifacts={str(p.relative_to(root)):digest(p) for p in sorted(root.rglob('*'))
                       if p.is_file() and p.suffix in ('.json','.jsonl','.log')}))
    assert {r['threshold'] for r in records}=={256,512,1024}
    for key in ('dev_sha256','profile_sha256','profile'):
        assert len({r['protocol'][key] for r in records})==1,f'PROTOCOL_DRIFT:{key}'
    output.mkdir(parents=True,exist_ok=False)
    for (threshold,root,_),record in zip(sources,records):
        dest=output/f'threshold-{threshold}';dest.mkdir()
        (dest/'summary.json').write_bytes((root/'summary.json').read_bytes())
    (output/'COLLECTION_PROVENANCE.json').write_text(json.dumps(dict(
        status='SYSTEM_PROBES_PASS',timing_scope='PARALLEL_NOT_THRESHOLD_SELECTION',
        source_records=records),indent=2)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',nargs=3,action='append',required=True,
        metavar=('THRESHOLD','ROOT','PROVENANCE'))
    p.add_argument('--reference-provenance',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    collect([(int(t),Path(r),Path(v)) for t,r,v in a.source],a.output,a.reference_provenance)


if __name__=='__main__':main()
