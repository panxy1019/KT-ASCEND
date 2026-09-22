import unittest
from migration_bridge.round9.runtime.shadow_decomposition import Recorder, phase, activate, summarize
class ShadowDecompositionTests(unittest.TestCase):
 def test_off_never_records(self):
  calls=[]
  with phase('prepare'):calls.append('compute')
  self.assertEqual(calls,['compute'])
 def test_failed_event_preserves_provider_exception(self):
  def bad():raise OSError('observer')
  r=Recorder(device_factory=bad)
  with self.assertRaisesRegex(ValueError,'provider'):
   with activate(r):
    with phase('prepare',device=True):raise ValueError('provider')
  self.assertTrue(r.errors);self.assertEqual(r.events[0]['outcome'],'error')
 def test_nested_union_not_double_counted(self):
  rows=[dict(id=1,parent=None,name='prepare',start_ns=0,end_ns=100),dict(id=2,parent=1,name='read',start_ns=10,end_ns=60),dict(id=3,parent=1,name='copy',start_ns=40,end_ns=90)]
  s=summarize(rows);self.assertEqual(s['parents'][0]['covered_ns'],80);self.assertEqual(s['parents'][0]['residual_ns'],20)
 def test_cross_boundary_rejected_not_clipped(self):
  rows=[dict(id=1,parent=None,name='prepare',start_ns=5,end_ns=100),dict(id=2,parent=1,name='read',start_ns=0,end_ns=10)]
  self.assertEqual(summarize(rows)['status'],'INVALID')
 def test_empty_trace_is_incomplete(self):self.assertEqual(summarize([])['status'],'INCOMPLETE')
 def test_bounded_trace_downgrades(self):
  r=Recorder(max_events=1)
  with activate(r):
   with phase('prepare'):
    with phase('read'):pass
  self.assertGreater(r.dropped,0);self.assertEqual(r.report()['status'],'INCOMPLETE')
 def test_device_failure_cannot_replace_return(self):
  class Event:
   def record(self):raise RuntimeError('event')
  r=Recorder(device_factory=lambda:(Event(),Event()))
  with activate(r):
   with phase('prepare',device=True):value=42
  self.assertEqual(value,42);self.assertTrue(r.errors)
 def test_identity_metadata_and_parent(self):
  r=Recorder()
  with activate(r):
   with phase('prepare',layer=0,moves=((0,20),(1,150))):
    with phase('source_read',projection='gate'):pass
  child=next(x for x in r.events if x['name']=='source_read');parent=next(x for x in r.events if x['name']=='prepare')
  self.assertEqual(child['parent'],parent['id']);self.assertEqual(parent['metadata']['layer'],0)
if __name__=='__main__':unittest.main()

class InstrumentationContractTests(unittest.TestCase):
 def test_real_copy_provider_has_named_boundaries(self):
  import inspect
  from migration_bridge.round9.runtime.dynamic_adapter import TorchKTEPAdapter
  source=inspect.getsource(TorchKTEPAdapter.copy_experts)
  for name in ['host_materialization','nd_allocation','copy_call','nz_index_copy','existing_sync','staging_retire']:
   self.assertIn(name,source)
 def test_no_compute_statement_change_after_stripping_observation(self):
  import ast,subprocess,pathlib
  class Strip(ast.NodeTransformer):
   def visit_Expr(self,node):
    if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=="record_tensor":return None
    return node
   def visit_ImportFrom(self,node):
    return None if node.module=='shadow_decomposition' else node
   def visit_With(self,node):
    node=self.generic_visit(node)
    if len(node.items)==1 and isinstance(node.items[0].context_expr,ast.Call) and isinstance(node.items[0].context_expr.func,ast.Name) and node.items[0].context_expr.func.id=='phase':return node.body
    return node
  for filename in ['dynamic_adapter.py','expert_source.py']:
   root=pathlib.Path(__file__).resolve().parents[1]
   before=(root/'tests/fixtures/shadow_baseline'/filename).read_text()
   current=(root/'runtime'/filename).read_text()
   self.assertEqual(ast.dump(ast.parse(before)),ast.dump(Strip().visit(ast.parse(current))))

class FixtureTests(unittest.TestCase):
 def test_derived_fixture_changes_one_expert_without_mutating_capture(self):
  from migration_bridge.round9.tests.shadow_decomposition_probe import derive_routes
  a=[20,150,14,111,203,414,211,358,33,230];before=a.copy();b=derive_routes(a)
  self.assertEqual(a,before);self.assertEqual(set(b)-set(a),{231});self.assertEqual(set(a)-set(b),{230});self.assertEqual(len(set(b)),10)
 def test_unrecognized_capture_rejected(self):
  from migration_bridge.round9.tests.shadow_decomposition_probe import derive_routes
  with self.assertRaises(ValueError):derive_routes(list(range(10)))

class ReviewRegressionTests(unittest.TestCase):
 def test_off_on_identity_must_not_change(self):
  from migration_bridge.round9.tools.launch_shadow_decomposition import validate_pair_identity
  old=dict(manifest_sha256='m',capture_sha256='c',capture_meta_sha256='j')
  for key in old:
   changed=dict(old);changed[key]='changed'
   with self.assertRaises(ValueError):validate_pair_identity(old,changed)
 def test_phase_costs_and_device_are_reported_separately(self):
  rows=[dict(id=1,parent=None,name='prepare',start_ns=0,end_ns=100),dict(id=2,parent=1,name='host_materialization',start_ns=10,end_ns=60),dict(id=3,parent=2,name='source_file_read',start_ns=20,end_ns=50),dict(id=4,parent=1,name='copy_call',start_ns=60,end_ns=90,device_ms=1.25)]
  s=summarize(rows);self.assertEqual(s['phases']['source_file_read']['host_union_ns'],30);self.assertEqual(s['phases']['copy_call']['device_ms'],[1.25]);self.assertEqual(s['parents'][0]['covered_ns'],80)
 def test_errored_provider_does_not_make_valid_report(self):
  r=Recorder()
  with self.assertRaises(ValueError):
   with activate(r):
    with phase('prepare'):raise ValueError('provider')
  self.assertEqual(r.report()['status'],'INCOMPLETE')
 def test_context_correlation_in_every_nested_event(self):
  r=Recorder(run_id='run')
  r.context=dict(case='B',request_id='req',allocation_epoch=2,placement_epoch=0,shadow_lifetime_id='shadow-2')
  with activate(r):
   with phase('prepare'):
    with phase('copy_call'):pass
  self.assertTrue(all(e['run_id']=='run' and e['context']['case']=='B' for e in r.events))
 def test_physical_off_on_results_must_match(self):
  from migration_bridge.round9.tests.shadow_decomposition_probe import compare_modes
  off=dict(copy_calls=2,copied_logical_bytes=100,cases=[dict(name='A',output_sha256='a'),dict(name='B',output_sha256='b')])
  on=dict(off);on['cases']=[dict(name='A',output_sha256='changed'),dict(name='B',output_sha256='b')]
  with self.assertRaises(ValueError):compare_modes(off,on)

class ResourceArithmeticTests(unittest.TestCase):
 def test_execution_peak_includes_projection_and_workspace(self):
  from migration_bridge.round9.tests.shadow_decomposition_probe import validate_execution_peak
  validate_execution_peak(616490496,209715200,512*1024**2)
  with self.assertRaises(ValueError):validate_execution_peak(209715200+512*1024**2+1,209715200,512*1024**2)

class TensorIdentityTests(unittest.TestCase):
 def test_tensor_identity_is_read_only_and_bounded(self):
  from migration_bridge.round9.runtime.shadow_decomposition import record_tensor
  class Storage:
   def data_ptr(self):return 123
   def nbytes(self):return 16
  class Tensor:
   shape=(2,4);dtype='bf16';device='cpu'
   def stride(self):return (4,1)
   def data_ptr(self):return 123
   def storage_offset(self):return 0
   def untyped_storage(self):return Storage()
  r=Recorder()
  with activate(r):
   with phase('nd_allocation'):record_tensor('staged',Tensor(),layout=2)
  self.assertEqual(r.events[0]['metadata']['tensors']['staged']['storage_pointer'],123)
