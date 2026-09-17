"""Verify full coverage and C32 byte parity against untouched product evidence."""
import argparse
import json
from pathlib import Path
from common import sha,write

p=argparse.ArgumentParser();p.add_argument('--matrix',type=Path,required=True);a=p.parse_args()
s=json.loads((a.matrix/'summary.json').read_text());assert s['status']=='PASS'
baseline=Path('/workspace/results/r65-r67-serial-v5/R16/operators')
decode=0;prefill=0;weight_experts=set();maximum=0
for layer in range(1,27):
    old=json.loads((baseline/f'L{layer:02d}.json').read_text())
    old_rows={(r['expert'],r['call']):r for r in old['rows']}
    reports=[json.loads((a.matrix/f'tp{tp}-L{layer:02d}/result.json').read_text()) for tp in (1,2)]
    assert all(r['status']=='PASS' for r in reports)
    assert reports[1]['weight_identity']=='EXACT'
    weight_experts.add(reports[1]['weight_identity_expert'])
    for r in reports[0]['rows']:
        if r['case'].startswith('real'):
            ref=old_rows[r['expert'],(1,4,7)[int(r['case'][4:])]]
            assert r['input_sha256']==ref['input_tensor_sha256'],'INPUT_DRIFT'
            assert set(ref['hashes'])=={r['output_sha256']},'C32_OUTPUT_REGRESSION'
    for r in reports[1]['rows']:
        assert r['repeat_exact'] and r['merge_exact'] and r['cpu_not_hit_exact']
        maximum=max(maximum,*r['metrics'].values())
        if r['case'].startswith('real'):decode+=1
        else:prefill+=1
assert decode==156 and prefill==64 and maximum<=.01
assert weight_experts=={1,63},'MULTI_EXPERT_WEIGHT_RECONSTRUCTION'
write(a.matrix/'audit.json',dict(status='PASS',decode_cases=decode,prefill_cases=prefill,
    c32_exact_vs_original_cases=decode,weight_reconstruction_layers=26,
    weight_reconstruction_experts=sorted(weight_experts),max_rel_l2=maximum,
    repeats=10,summary_sha256=sha(a.matrix/'summary.json')))
