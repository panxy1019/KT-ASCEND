"""Freeze exact existing assets and new product protocol before any R6.7 run."""
import argparse
import json
from pathlib import Path
import shutil
from common import sha, validate_profile, write


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if not str(a.output).startswith('/workspace/results/r65-r67-'):
        raise ValueError('STUDY_SCOPE')
    base = Path('/workspace/results')
    quality = base / 'r65-r66b-base-protocol-freeze-v1/quality.json'
    if sha(quality) != '885ccf76386dcfed2867142417300ac69c4401754360f92e6b6e4f179adffd97':
        raise ValueError('PARENT_QUALITY_DRIFT')
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output / 'profiles').mkdir()
    manifests = {}
    for n in (64, 32, 16, 8):
        source = base / f'r65-profiles-v1/all-R{n}.json'
        data = json.loads(source.read_text()); validate_profile(data)
        if sha(source.with_suffix('.pt')) != data['pt_sha256']:
            raise ValueError('PLACEMENT_BYTES_DRIFT')
        for suffix in ('.json', '.pt'):
            shutil.copyfile(source.with_suffix(suffix), a.output / 'profiles' / f'R{n}{suffix}')
        manifests[f'R{n}'] = dict(profile_sha256=sha(source), canonical_sha256=data['canonical_sha256'],
                                 pt_sha256=data['pt_sha256'])
    raw = json.loads(quality.read_text())
    rows = [row for row in raw['rows'] if row['benchmark'] == 'cmmlu']
    if len(rows) != 256 or len({row['id'] for row in rows}) != 256:
        raise ValueError('CMMLU_SCOPE')
    write(a.output / 'cmmlu.json', dict(version='r67-cmmlu256-5shot-v1', rows=rows,
          parent_manifest_sha256=sha(quality), selection='EXACT_R66B_CMMLU_ROWS_UNCHANGED',
          protocol='5shot_conditional_letter_likelihood_ABCD_fixed_tie_order', seed=0))
    corpus = base / 'corpora-v2/s6.json'
    prompts = {row['id']: row for row in json.loads(corpus.read_text())['prompts']}
    compact = [prompts[k] for k in ('s6_en_01', 's6_zh_01', 's6_math_01', 's6_json_01')]
    write(a.output / 'compact.json', dict(version='r67-compact-v1', prompts=compact,
          parent_sha256=sha(corpus), budgets=[16, 8, 8, 8, 8, 8, 4, 2], restart_budget=16,
          observer_budget=8, operator_calls=[1, 4, 7], operator_experts=[1, 63]))
    runtime = base / 'r65-r66-runtime-source-final-b0.json'
    if sha(runtime) != '68d1a5814a70e123eb55c840ec9a995fda1fe4802120374f61f32e2d1775570a':
        raise ValueError('RUNTIME_MANIFEST_DRIFT')
    for item in json.loads(runtime.read_text())['files']:
        if sha(item['path']) != item['sha256']:
            raise ValueError('IMPORTED_RUNTIME_DRIFT:' + item['path'])
    assets = json.loads((base / 'r65-r66-quality-execution-approval-v1.json').read_text())['asset_sha256']
    for path, digest in assets.items():
        if sha(path) != digest: raise ValueError('ASSET_PROVENANCE:' + path)
    gguf = base / 'r65-assets-v1/all-moe-f32.gguf'
    if gguf.stat().st_size != 57579411648: raise ValueError('GGUF_SIZE')
    identity = json.loads((gguf.parent / 'identity.json').read_text())
    if identity['status'] != 'ALL_EXPERT_WEIGHT_IDENTITY_PASS' or identity['gguf_sha256'] != '20114dba7790a9b6be40a17ffb3881c516b81ea86b8f2fcfd3ea61d052e96fe2':
        raise ValueError('GGUF_IDENTITY')
    external = ['/workspace/r65-tools/migration_bridge/round6_5/tools/owned_guard.py',
                '/workspace/r65-tools/migration_bridge/round6_5/tools/r65_structure_hook.py',
                '/workspace/results/round6_2-prequalification/tools/run_owned_process.py',
                '/workspace/results/r64-migration-tools-v6/tools/stream_timing.py',
                '/workspace/results/r64-migration-tools-v6/tools/run_compact_controls.py']
    write(a.output / 'STUDY_FREEZE.json', dict(version='round6.7-serial-v1',
          profile_order=['R64', 'R32', 'R16', 'R8'], profiles=manifests,
          cmmlu_sha256=sha(a.output / 'cmmlu.json'), compact_sha256=sha(a.output / 'compact.json'),
          runtime_manifest=str(runtime), runtime_manifest_sha256=sha(runtime),
          tool_hashes={p.name:sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
          shell_sha256=sha(Path(__file__).with_name('launch_serial.sh')),
          external_tool_hashes={p:sha(p) for p in external}, asset_sha256=assets,
          gguf_stat=dict(size=gguf.stat().st_size, mtime_ns=gguf.stat().st_mtime_ns),
          local_audit_base='93844959b32f64a37b70346bd5c7e0df2394f5d4',
          container='kt-r65-static-offload', npu=0, cpu_set='0-31', workers=32,
          batch=1, tp=1, deferred=0, dynamic=False, graph=False,
          quality_drop_pp=5.0, population_noninferiority_claim=False,
          gguf=str(base / 'r65-assets-v1/all-moe-f32.gguf'),
          gguf_sha256='20114dba7790a9b6be40a17ffb3881c516b81ea86b8f2fcfd3ea61d052e96fe2',
          performance=dict(budgets=32, warmups=2, timed=3,
                           prompt_ids=['s6_en_01', 's6_json_01'], streaming=True,
                           hard_throughput_floor=None), operator_rel_l2_max=.01))
    print(json.dumps(dict(status='STUDY_FROZEN', path=str(a.output),
                         sha256=sha(a.output / 'STUDY_FREEZE.json'))))


if __name__ == '__main__': main()
