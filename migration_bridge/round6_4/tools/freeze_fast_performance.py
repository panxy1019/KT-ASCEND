"""Derive fixed single-shape three-block manifest before any formal timing."""
import hashlib
import json
from pathlib import Path
from run_compact_controls import checked
from freeze_corpora import canonical_sha


def main():
    fixtures=Path(__file__).parent.parent/'fixtures'
    source=fixtures/'performance.json'
    spec=checked(source)
    spec.pop('sha256')
    spec.update(version='r64-performance-fast-v1',paired_blocks=3,
        shapes=[r for r in spec['shapes'] if r['id']==spec['primary_shape']],
        confidence_method='two_sided_95_student_t_on_three_independent_log_block_ratios',
        parent_raw_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    spec['sha256']=canonical_sha(spec)
    target=fixtures/'performance-fast.json'
    with target.open('x') as f:
        json.dump(spec,f,ensure_ascii=False,indent=2)
        f.write('\n')
    print(hashlib.sha256(target.read_bytes()).hexdigest())


if __name__=='__main__':
    main()
