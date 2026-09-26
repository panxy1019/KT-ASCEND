import unittest
from unittest.mock import patch
from types import SimpleNamespace as NS
from migration_bridge.round9.runtime.dynamic import PlacementController
from migration_bridge.round9.runtime.dynamic_boundary import RequestBoundary, scheduler_cancel
class Adapter:
    def synchronize(self):pass
    def copy_expert(self,*args):pass
    def publish(self,*args):pass
class BoundaryTests(unittest.TestCase):
    def setUp(self):
        self.controller=PlacementController({0:(0,1)},Adapter(),experts=4,resident=2)
        self.boundary=RequestBoundary(self.controller)
        self.req=NS(rid='one',req_pool_idx=3,mamba_pool_idx=1,done=False)
        self.req.finished=lambda:self.req.done
    def test_chunks_keep_one_generation_and_cleanup_precedes_commit(self):
        b=self.boundary;b.before([self.req]);b.observe(0,[2,2,1]);b.after()
        self.assertEqual(self.controller.generation,0)
        b.before([self.req]);self.req.done=True
        with self.assertRaisesRegex(RuntimeError,'released'):b.after()
        self.req.req_pool_idx=self.req.mamba_pool_idx=None;b.after()
        self.assertEqual(self.controller.generation,1)
        self.assertIsNone(b.request)
    def test_unleased_or_interleaved_requests_rejected(self):
        with self.assertRaises(RuntimeError):self.boundary.observe(0,[1])
        self.boundary.before([self.req])
        with self.assertRaises(RuntimeError):self.boundary.before([NS(rid='other')])
        with self.assertRaises(RuntimeError):self.boundary.before([self.req,self.req])
    def test_abort_uses_the_same_drained_boundary(self):
        self.boundary.before([self.req]);self.boundary.observe(0,[3,3,1])
        self.req.done=True;self.req.req_pool_idx=self.req.mamba_pool_idx=None
        self.boundary.after();self.assertEqual(self.controller.generation,1)
    def test_explicit_cancel_discards_routes_without_migration(self):
        self.boundary.before([self.req]);self.boundary.observe(0,[3,3,1])
        self.req.done=True;self.req.req_pool_idx=self.req.mamba_pool_idx=None
        scheduler=NS(_round9_dynamic_boundary=self.boundary)
        with patch(
                'migration_bridge.round9.runtime.dynamic_boundary._enabled',
                return_value=True):
            scheduler_cancel(scheduler,self.req)
        self.assertEqual(self.controller.generation,0)
        self.assertEqual(self.controller._counts,{})
        self.assertIsNone(self.boundary.request)
class ProfileBindingTests(unittest.TestCase):
    def test_dynamic_rejects_missing_static_changed_and_partial_model_profiles(self):
        import tempfile,json
        from pathlib import Path
        from unittest.mock import patch
        from migration_bridge.round9.runtime import candidate_config as config
        with tempfile.TemporaryDirectory() as directory,patch.object(config,'_MODEL_BINDING',None),patch.object(config,'_MODEL_LAYERS',set()):
            path=Path(directory)/'profile.json'
            with self.assertRaises(RuntimeError):config.require_dynamic_model_profile(None)
            with self.assertRaises(FileNotFoundError):config.require_dynamic_model_profile(path)
            doc=dict(workers=128,storage='Q8_0',placement='static',numa_nodes=[2,3,6,7])
            path.write_text(json.dumps(doc))
            with self.assertRaisesRegex(RuntimeError,'request_boundary'):config.require_dynamic_model_profile(path)
            doc['placement']='request_boundary';path.write_text(json.dumps(doc))
            config.bind_model_profile(path,0,{'catalog_sha256':'test'})
            with self.assertRaisesRegex(RuntimeError,'all48'):config.require_dynamic_model_profile(path)
            for layer in range(1,48):config.bind_model_profile(path,layer,{'catalog_sha256':'test'})
            self.assertIn('profile_sha256',config.require_dynamic_model_profile(path))
            doc['storage']='MXFP4';path.write_text(json.dumps(doc))
            with self.assertRaisesRegex(RuntimeError,'all48'):config.require_dynamic_model_profile(path)
            with self.assertRaisesRegex(RuntimeError,'changed'):config.bind_model_profile(path,1,{'catalog_sha256':'test'})

class DeviceGateTests(unittest.TestCase):
    def test_nz_gate_binds_storage_workers_source_and_exact_migrations(self):
        import tempfile,json,hashlib
        from pathlib import Path
        from migration_bridge.round9.runtime.candidate_config import require_dynamic_device_gate as check
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'gate.json'
            g=dict(status='PASS_REAL_KTEP_LOCAL_MERGE',storage='BF16',cpu_workers=128,source_identity={'revision':'test'},npu_weight_formats={'w13_weight':29,'w2_weight':29},dynamic_migrations=[dict(all64_processed_experts_exact=True,pinned_mask_pointer_stable=True,formats_after={'w13_weight':29,'w2_weight':29})]*3)
            g['bank_sha256']='bank-test'
            g['batch_copy_probe']=dict(experts=64,reverse_and_restore_exact=True,max_device_staging_bytes=400*1024**2)
            g['dynamic_migrations']=[dict(g['dynamic_migrations'][0],generation=i+1,moves=[[0,e]]) for i,e in enumerate((511,0,511))]
            g['rows']=[dict(tokens=q,kind=k,errors=dict(cpu_partial=0.,npu_owned_partial=0.,routed_merge=0.),hashes=['same']*3) for q in (1,10,32) for k in ('mixed','resident_only','repeated_cpu')]
            def run():
                p.write_text(json.dumps(g));return check(p,hashlib.sha256(p.read_bytes()).hexdigest(),storage='BF16',workers=128,source_identity={'revision':'test'},layer0_sha256='bank-test')
            self.assertEqual(run()['status'],g['status'])
            for key,value in [('batch_copy_probe',{}),('bank_sha256','wrong'),('rows',[]),('storage','Q8_0'),('cpu_workers',64),('source_identity',{}),('npu_weight_formats',{'w13_weight':2,'w2_weight':2}),('dynamic_migrations',[])]:
                old=g[key];g[key]=value
                with self.assertRaises(RuntimeError):run()
                g[key]=old
            with self.assertRaises(RuntimeError):check(None,None,storage='BF16',workers=128,source_identity={},layer0_sha256='bank-test')

if __name__=='__main__':unittest.main()
