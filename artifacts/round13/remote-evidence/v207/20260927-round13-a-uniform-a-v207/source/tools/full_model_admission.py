"""Pure full-model budget evaluator; never launches a model or reserves resources.

The caller supplies independently verified gate receipts. Hash fields identify
those receipts; this evaluator does not substitute for verifying their contents.
A PASS expires with the underlying telemetry and is not model qualification.
"""
import argparse
import collections
import importlib.util
import json
import math
import pathlib
import re
import time

_spec=importlib.util.spec_from_file_location('round9_base_admission',pathlib.Path(__file__).with_name('admission.py'))
_base=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)
REVISION='de4b8e4d43b917e7706784d8bb445c9af86a3540'
PROFILE='Q64_BF16_C64_SYNC_TEXT_TP1'
REQUIRED_GATES=('source_identity','source_semantics','runtime_integration','cpu_c64',
                'qsa','gdn','gr','ple_sync','ktep_ownership','layout_and_scaling')
HBM_COMPONENTS=('routed_resident','other_weights','kv_states','runtime_workspace','load_temporary')
BUDGET_BASES=('STATIC_FIRST_RUN_BOUND','MEASURED_FULL_WORKLOAD_PEAK')


def evaluate(snapshot, evidence, now):
    if not isinstance(snapshot,dict):
        return {'status':'BLOCKED','reasons':['missing device snapshot']}
    result=_base.evaluate(snapshot,now,'a3-physical15')
    errors=list(result['reasons'])
    if not isinstance(evidence,dict):
        return {'status':'BLOCKED','reasons':errors+['missing full-model evidence']}
    if evidence.get('profile')!=PROFILE:
        errors.append('full-model reference profile is not frozen')
    def digest(v):
        return isinstance(v,str) and re.fullmatch('[0-9a-f]{64}',v) is not None
    implementation=evidence.get('implementation_sha256')
    if not digest(implementation):
        errors.append('runtime implementation identity is missing')
    def receipt_bound(r):
        return (isinstance(r,dict) and r.get('status')=='PASS'
                and digest(r.get('artifact_sha256'))
                and r.get('profile')==PROFILE and r.get('source_revision')==REVISION
                and r.get('implementation_sha256')==implementation)
    gates=evidence.get('gates',{})
    for key in REQUIRED_GATES:
        receipt=gates.get(key) if isinstance(gates,dict) else None
        if not receipt_bound(receipt) or receipt.get('gate')!=key:
            errors.append('missing verified gate: '+key)
    if evidence.get('training_peak_bounded') is not True:
        errors.append('training future peak is not bounded; current idle snapshot insufficient')
    def valid(v):
        return type(v) in (int,float) and math.isfinite(v) and v>0
    bound=evidence.get('training_bound')
    if (not isinstance(bound,dict) or not digest(bound.get('artifact_sha256'))
            or bound.get('host')!='110.123.0.3' or bound.get('physical_id')!=15
            or bound.get('basis') not in ('ENFORCED_RESERVATION','VERIFIED_WORKLOAD_ENVELOPE',
                                          'USER_ATTESTED_WORKLOAD_ENVELOPE')
            or not valid(bound.get('valid_until')) or bound['valid_until']<=now
            or not valid(bound.get('peak_mib'))
            or bound['peak_mib']!=snapshot.get('training_peak_mib')):
        errors.append('training bound receipt missing, expired or unrelated to device/peak')
    envelope=evidence.get('budget_envelope')
    workload=('context_tokens','max_new_tokens','batch_size','max_inflight')
    if (not all(type(evidence.get(k)) is int and evidence[k]>0 for k in workload)
            or evidence.get('batch_size')!=1 or evidence.get('max_inflight')!=1):
        errors.append('missing or unsupported workload envelope')
    elif evidence['max_new_tokens']>=evidence['context_tokens']:
        errors.append('output reserve must be smaller than total context')
    bound_fields=workload+('host_owned_budget_mib','disk_required_mib','hbm_components_mib')
    if (not receipt_bound(envelope)
            or any(envelope.get(k)!=evidence.get(k) for k in bound_fields)):
        errors.append('resource budget is not bound to verified workload/implementation envelope')
    budget_basis=envelope.get('basis') if isinstance(envelope,dict) else None
    if budget_basis not in BUDGET_BASES:
        errors.append('full-model budget basis is missing or unsupported')
    fields=('host_available_mib','host_owned_budget_mib','host_reserve_mib',
            'disk_free_mib','disk_required_mib')
    if not all(valid(evidence.get(k)) for k in fields):
        errors.append('missing or invalid host/disk budget')
    else:
        if not 655360 <= evidence['host_owned_budget_mib'] <= 786432:
            errors.append('reference host budget must be640–768GiB')
        if evidence['host_reserve_mib']<65536 or evidence['disk_required_mib']<1258292:
            errors.append('initial policy requires64GiB host reserve and1.2TiB workspace')
        if evidence['host_owned_budget_mib']+evidence['host_reserve_mib']>evidence['host_available_mib']:
            errors.append('insufficient host memory with reserve')
        if evidence['disk_required_mib']>evidence['disk_free_mib']:
            errors.append('insufficient working disk space')
    components=evidence.get('hbm_components_mib')
    if (not isinstance(components,dict) or set(components)!=set(HBM_COMPONENTS)
            or not all(valid(components.get(k)) for k in HBM_COMPONENTS)):
        errors.append('HBM budget must include all loading and steady-state components')
    else:
        if components['routed_resident']!=28800:
            errors.append('Q64 BF16 routed payload must be28800MiB')
        owned=snapshot.get('owned_peak_budget_mib') if isinstance(snapshot,dict) else None
        if not valid(owned) or sum(components.values())>owned:
            errors.append('HBM components exceed owned peak budget')
    cpus=snapshot.get('cpu_set')
    visible=snapshot.get('visible_cpu_set')
    numa=snapshot.get('visible_cpu_numa')
    topology=snapshot.get('visible_cpu_topology')
    expected_visible=list(range(480,640))
    expected_numa={'6':list(range(480,560)),'7':list(range(560,640))}
    topology_ok=(isinstance(topology,dict)
                 and set(topology)=={str(cpu) for cpu in expected_visible}
                 and all(isinstance(key,str) and key for key in topology.values()))
    if topology_ok:
        groups=collections.defaultdict(list)
        for cpu in expected_visible:
            groups[topology[str(cpu)]].append(cpu)
        topology_ok=(len(groups)==80 and all(len(group)==2 for group in groups.values())
                     and all(all((cpu < 560)==(group[0] < 560) for cpu in group)
                             for group in groups.values()))
    if visible!=expected_visible or numa!=expected_numa or not topology_ok:
        errors.append('visible CPU set must be480-639 with80 SMT2 cores on NUMA6/7')
    quota=snapshot.get('cpu_quota_cores')
    workers=snapshot.get('worker_physical_cores')
    auxiliary=snapshot.get('auxiliary_core_budget')
    if (type(quota) is not int or type(workers) is not int or type(auxiliary) is not int
            or quota!=72 or workers!=64 or auxiliary!=8 or workers+auxiliary!=quota):
        errors.append('C64 reference requires72-core quota:64workers plus8auxiliary')
    selected_ok=(isinstance(cpus,list) and len(cpus)==72
                 and all(type(cpu) is int and cpu in expected_visible for cpu in cpus)
                 and topology_ok
                 and snapshot.get('cpu_physical_keys')==[topology[str(cpu)] for cpu in cpus]
                 and snapshot.get('observed_cpu_keys')=={
                     str(cpu):topology[str(cpu)] for cpu in cpus})
    if not selected_ok:
        errors.append('selected worker/aux CPUs do not match visible physical-core topology')
    if errors:
        return {'status':'BLOCKED','scope':'FULL_MODEL_RESOURCE_PREFLIGHT_ONLY',
                'full_model_peak_status':'NOT_MEASURED','reasons':errors}
    if budget_basis=='STATIC_FIRST_RUN_BOUND':
        return {'status':'STATIC_BOUND_READY_FOR_MONITORED_FULL_MODEL_RUN',
                'scope':'MONITORED_FULL_MODEL_FIRST_RUN_PREFLIGHT',
                'full_model_peak_status':'NOT_MEASURED','reasons':[]}
    return {'status':'PASS','scope':'FULL_MODEL_RESOURCE_PREFLIGHT_ONLY',
            'full_model_peak_status':'MEASURED','reasons':[]}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('snapshot',type=pathlib.Path)
    p.add_argument('evidence',type=pathlib.Path)
    a=p.parse_args()
    result=evaluate(json.loads(a.snapshot.read_text()),json.loads(a.evidence.read_text()),time.time())
    print(json.dumps(result,indent=2))
    ready=('PASS','STATIC_BOUND_READY_FOR_MONITORED_FULL_MODEL_RUN')
    raise SystemExit(0 if result['status'] in ready else 2)

if __name__=='__main__': main()
