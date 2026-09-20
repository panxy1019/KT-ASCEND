"""Full-model admission must not inherit the bounded runtime probe's PASS."""
import copy
import importlib.util
import pathlib
import unittest

P=pathlib.Path(__file__).parents[1]/'tools/full_model_admission.py'
class FullModelAdmissionTests(unittest.TestCase):
    def module(self):
        spec=importlib.util.spec_from_file_location('full_model_admission',P)
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        return mod

    def snapshot(self):
        visible=list(range(480,640))
        topology={str(cpu):f'numa{6 if cpu < 560 else 7}-core{(cpu % 80)//2}'
                  for cpu in visible}
        selected=list(range(480,560,2))+list(range(560,624,2))
        return dict(host='110.123.0.3',pci_bus_id='0000:83:00.0',logical_device=0,
            physical_id=15,device_nodes=[15],mapping_verified=True,access_verified=True,
            identity_status='PASS',prerequisite_statuses={'runtime':'PASS'},
            sampled_at=1000,hbm_total_mib=65536,training_peak_mib=10000,
            owned_peak_budget_mib=42000,reserve_mib=12288,telemetry_ok=True,
            cpu_set=selected,cpu_physical_keys=[topology[str(cpu)] for cpu in selected],
            observed_cpu_keys={str(cpu):topology[str(cpu)] for cpu in selected},
            visible_cpu_set=visible,
            visible_cpu_numa={'6':list(range(480,560)),'7':list(range(560,640))},
            visible_cpu_topology=topology,cpu_quota_cores=72,
            worker_physical_cores=64,auxiliary_core_budget=8)

    def evidence(self,mod):
        return dict(profile='Q64_BF16_C64_SYNC_TEXT_TP1',
            implementation_sha256='b'*64,context_tokens=4096,max_new_tokens=256,batch_size=1,max_inflight=1,
            gates={k:{'gate':k,'status':'PASS','artifact_sha256':'a'*64,'source_revision':mod.REVISION,'profile':mod.PROFILE,'implementation_sha256':'b'*64} for k in mod.REQUIRED_GATES},
            budget_envelope={'status':'PASS','basis':'MEASURED_FULL_WORKLOAD_PEAK','artifact_sha256':'c'*64,'profile':mod.PROFILE,'source_revision':mod.REVISION,'implementation_sha256':'b'*64,'context_tokens':4096,'max_new_tokens':256,'batch_size':1,'max_inflight':1,'host_owned_budget_mib':786432,'disk_required_mib':1258292,'hbm_components_mib':{'routed_resident':28800,'other_weights':6000,'kv_states':2000,'runtime_workspace':3000,'load_temporary':2000}},
            training_bound={'artifact_sha256':'d'*64,'host':'110.123.0.3','physical_id':15,'peak_mib':10000,'valid_until':1100,'basis':'ENFORCED_RESERVATION'},
            training_peak_bounded=True,
            host_available_mib=1900000,host_owned_budget_mib=786432,host_reserve_mib=65536,
            disk_free_mib=2000000,disk_required_mib=1258292,
            hbm_components_mib={'routed_resident':28800,'other_weights':6000,'kv_states':2000,'runtime_workspace':3000,'load_temporary':2000})

    def test_runtime_only_cannot_admit_full_model(self):
        mod=self.module()
        self.assertEqual(mod.evaluate(self.snapshot(),{},1000)['status'],'BLOCKED')

    def test_complete_evidence_and_each_missing_gate(self):
        mod=self.module();e=self.evidence(mod)
        self.assertEqual(mod.evaluate(self.snapshot(),e,1000)['status'],'PASS')
        for key in mod.REQUIRED_GATES:
            x=copy.deepcopy(e);del x['gates'][key]
            with self.subTest(key=key): self.assertEqual(mod.evaluate(self.snapshot(),x,1000)['status'],'BLOCKED')

    def test_resource_and_stale_evidence_rejected(self):
        mod=self.module();e=self.evidence(mod)
        cases=[{'training_peak_bounded':False},{'host_available_mib':100000},
               {'disk_free_mib':1000},{'profile':'Q32'},
               {'hbm_components_mib':{'routed_resident':28800}},
               {'host_reserve_mib':float('nan')}]
        for change in cases:
            x=copy.deepcopy(e);x.update(change)
            with self.subTest(change=change): self.assertEqual(mod.evaluate(self.snapshot(),x,1000)['status'],'BLOCKED')
        self.assertEqual(mod.evaluate(self.snapshot(),e,1031)['status'],'BLOCKED')
        e['gates']['qsa']['artifact_sha256']='not-a-hash'
        self.assertEqual(mod.evaluate(self.snapshot(),e,1000)['status'],'BLOCKED')

    def test_malformed_inputs_fail_closed(self):
        mod=self.module();e=self.evidence(mod)
        x=copy.deepcopy(e);x['hbm_components_mib']['unexpected']='bad'
        self.assertEqual(mod.evaluate(self.snapshot(),x,1000)['status'],'BLOCKED')
        s=self.snapshot();s['cpu_set']=None
        self.assertEqual(mod.evaluate(s,e,1000)['status'],'BLOCKED')
        s=self.snapshot();s['cpu_set'][0]=True
        self.assertEqual(mod.evaluate(s,e,1000)['status'],'BLOCKED')
        s=self.snapshot();s['cpu_set'][0]=480.0
        self.assertEqual(mod.evaluate(s,e,1000)['status'],'BLOCKED')

    def test_real_visible_cpuset_and_physical_cpu_budget(self):
        mod=self.module();e=self.evidence(mod);s=self.snapshot()
        self.assertEqual(len(s['visible_cpu_set']),160)
        self.assertEqual(len(set(s['visible_cpu_topology'].values())),80)
        self.assertEqual(mod.evaluate(s,e,1000)['status'],'PASS')
        mutations=[
            ('visible_cpu_set',list(range(480,639))),
            ('visible_cpu_numa',{'6':list(range(480,560)),'7':list(range(560,639))}),
            ('visible_cpu_topology',{**s['visible_cpu_topology'],'639':'numa7-core39-wrong'}),
            ('cpu_quota_cores',71),('worker_physical_cores',63),
            ('auxiliary_core_budget',7),
        ]
        for key,value in mutations:
            bad=copy.deepcopy(s);bad[key]=value
            with self.subTest(key=key):
                self.assertEqual(mod.evaluate(bad,e,1000)['status'],'BLOCKED')

    def test_static_budget_is_ready_only_for_monitored_first_run(self):
        mod=self.module();e=self.evidence(mod)
        e['budget_envelope']['basis']='STATIC_FIRST_RUN_BOUND'
        result=mod.evaluate(self.snapshot(),e,1000)
        self.assertEqual(result['status'],'STATIC_BOUND_READY_FOR_MONITORED_FULL_MODEL_RUN')
        self.assertEqual(result['scope'],'MONITORED_FULL_MODEL_FIRST_RUN_PREFLIGHT')
        self.assertEqual(result['full_model_peak_status'],'NOT_MEASURED')

    def test_budget_and_training_basis_fail_closed(self):
        mod=self.module();e=self.evidence(mod)
        e['training_bound']['basis']='USER_ATTESTED_WORKLOAD_ENVELOPE'
        self.assertEqual(mod.evaluate(self.snapshot(),e,1000)['status'],'PASS')
        for basis in ('ESTIMATE','',None):
            bad=copy.deepcopy(e);bad['budget_envelope']['basis']=basis
            with self.subTest(kind='budget',basis=basis):
                self.assertEqual(mod.evaluate(self.snapshot(),bad,1000)['status'],'BLOCKED')
        bad=copy.deepcopy(e);bad['training_bound']['basis']='VERIFIED_WORKLOAD_ENVELOPE'
        self.assertEqual(mod.evaluate(self.snapshot(),bad,1000)['status'],'PASS')
        bad['training_bound']['basis']='CURRENT_IDLE'
        self.assertEqual(mod.evaluate(self.snapshot(),bad,1000)['status'],'BLOCKED')

    def test_unbound_or_understated_full_model_evidence(self):
        mod=self.module();base=self.evidence(mod)
        mutations=[('host_owned_budget_mib',1),('host_reserve_mib',1),
                   ('disk_required_mib',1),('training_bound',{}),('context_tokens',1),
                   ('max_inflight',2),('batch_size',2),('max_new_tokens',128),('budget_envelope',{})]
        for key,value in mutations:
            e=copy.deepcopy(base);e[key]=value
            with self.subTest(key=key):self.assertEqual(mod.evaluate(self.snapshot(),e,1000)['status'],'BLOCKED')
        for key,value in [('gate','gr'),('source_revision','old'),('profile','Q32'),('implementation_sha256','e'*64)]:
            e=copy.deepcopy(base);e['gates']['qsa'][key]=value
            self.assertEqual(mod.evaluate(self.snapshot(),e,1000)['status'],'BLOCKED')
        for key,value in [('physical_id',14),('peak_mib',1),('valid_until',999),('basis','CURRENT_IDLE')]:
            e=copy.deepcopy(base);e['training_bound'][key]=value
            self.assertEqual(mod.evaluate(self.snapshot(),e,1000)['status'],'BLOCKED')
