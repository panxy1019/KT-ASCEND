"""Freeze DEV-selected configurations only after independent entry approval.

Creates artifact bytes, not a commit acknowledgement. The operator must commit
these bytes locally before creating/uploading freeze-ack.json.
"""
import hashlib
import json
from pathlib import Path

ROOT=Path('/home/admin/kt-round8-qwen35')
OUT=ROOT/'qwen-dynamic-candidate-v2'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')

def main():
    dev=ROOT/'q32-dynamic-transfer-v2';restart=ROOT/'q32-dynamic-restart-v2'
    assert json.loads((dev/'dynamic-dev-summary.json').read_text())['status']=='PASS'
    assert json.loads((restart/'summary.json').read_text())['status']=='PASS'
    for source in (dev,restart):assert 'Check input storage_offset' not in (source/'server.log').read_text()
    assert json.loads((restart/'state.json').read_text())['status']=='REPLAY_PASS_PENDING_INDEPENDENT_AUDIT'
    OUT.mkdir(exist_ok=False)
    audited=json.loads((dev/'resolved-command.json').read_text())
    plain=json.loads((restart/'resolved-command.json').read_text())
    configs={}
    for mode in ('heldout','cmmlu','perf-static','perf-dynamic'):
        cfg=json.loads(json.dumps(audited if mode=='heldout' else plain))
        nodes=[0,1] if mode=='heldout' else [2,3] if mode=='cmmlu' else [4,5]
        idx=cfg['argv'].index('--kt-numa-nodes');cfg['argv'][idx+1:idx+3]=list(map(str,nodes))
        cfg['numa_nodes']=nodes
        if mode=='heldout':
            idx=cfg['argv'].index('--forward-hooks')+1;hooks=json.loads(cfg['argv'][idx])
            hooks[0]['hook_factory']='qwen_dynamic_audit:make_structure'
            hooks[0]['config']['output']='/workspace/round8/qwen-dynamic-heldout-v2'
            cfg['argv'][idx]=json.dumps(hooks)
        else:assert '--forward-hooks' not in cfg['argv']
        if mode=='perf-static':
            cfg['argv'].remove('--kt-enable-dynamic-expert-update')
            idx=cfg['argv'].index('--kt-gpu-prefill-token-threshold')
            del cfg['argv'][idx:idx+2]
            cfg['dynamic']=False;cfg['dynamic_threshold']=0
        cfg['purpose']='FROZEN_DYNAMIC_QUALIFICATION_'+mode.upper()
        path=OUT/(mode+'.json');write(path,cfg);configs[mode]=sha(path)
    contract=dict(version='qwen35-q32-dynamic-v2',threshold=256,chunked_prefill_size=512,
        selection_reason='DEV-compatible trigger; medium256 formal workload exercises migration. No held-out or quality fitting.',
        dev1024='SKIPPED_BY_USER_AFTER_PRESERVED_CHUNK512_TRIGGER_FAILURE',
        model_revision='59d61f3ce65a6d9863b86d2e96597125219dc754',
        q32_profile_sha256='b358399e6405d60b81d8231529b1ec6024ff7138cddab2e0cbac3ccf0b25f486',
        cpu_manifest_sha256='35ca0ac63a3646ba7b89675053bbead19724325fa9d73f99910ef3c156a84405',
        cmmlu_manifest_sha256='365f557c0746db1e02ddee4ec66c7afb5e72eeb17e8dd8e0a47d1ed2b7af1b9d',
        heldout_sha256='67432eca56ee9638dfe72e1d670ef1e30ce8833fe6b457dead6e12a806bac2d7',
        configs_sha256=configs,quality_reference_correct=222,quality_min_correct=210,
        tools_sha256={name:sha(ROOT/'tools'/name) for name in (
            'qwen_dynamic_evaluation.py','run_qwen_dynamic_evaluation.py',
            'qwen_dynamic_perf.py','run_qwen_dynamic_perf.py',
            'qwen_dynamic_audit.py','qwen_static_perf.py','run_qwen_abba.py',
            'round8_resource_guard.py','admit_q64_host.py',
            'qwen_cmmlu_reference.py','audit_qwen_dynamic_results.py')},
        quality_count=256,heldout_count=128,heldout_decode_tokens=16,
        system_gate='PASS',performance=dict(short=[64,32],medium=[256,64],long=False,
            blocks=3,order=['static','dynamic','dynamic','static'],isolated=True,hooks=False),
        c64='2x32_SUBPOOLS',graph=False,deferred=False,batch=1,tp=1,
        evidence_sha256={'dev_summary':sha(dev/'dynamic-dev-summary.json'),
                         'restart_summary':sha(restart/'summary.json'),
                         'fixed_wrapper':sha(ROOT/'sglang-python-dynamic-v2/sglang/srt/layers/moe/kt_ep_wrapper.py')})
    assert sha(ROOT/'q32_profile.json')==contract['q32_profile_sha256']
    assert sha(ROOT/'cpu-experts-f32-v1/manifest.json')==contract['cpu_manifest_sha256']
    write(OUT/'candidate.json',contract)
    print(json.dumps(dict(candidate_sha256=sha(OUT/'candidate.json'),status='BYTES_PREPARED_COMMIT_REQUIRED')))

if __name__=='__main__':main()
