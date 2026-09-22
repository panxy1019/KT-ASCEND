"""Execute the installed real f02 event-loop method with bounded scheduler doubles.

No model/NPU calls; proves hook ordering for chunk/completion/abort, not serving.
"""
import argparse,ast,hashlib,json,os,pathlib
from types import SimpleNamespace as NS
from sglang.srt.layers.round9.dynamic_boundary import RequestBoundary
from sglang.srt.layers.round9.dynamic import PlacementController
p=argparse.ArgumentParser();p.add_argument('--scheduler',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args()
assert not a.output.exists();source=a.scheduler.read_text();tree=ast.parse(source)
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Scheduler')
method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='event_loop_normal');method.decorator_list=[]
ns={'envs':NS(SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY=NS(get=lambda:False))}
exec(compile(ast.Module([method],type_ignores=[]),str(a.scheduler),'exec'),ns)
class Done(Exception):pass
class Adapter:
 def synchronize(self):pass
 def copy_expert(self,*args):pass
 def publish(self,*args):pass
rows=[];os.environ['ROUND9_DYNAMIC_BOUNDARY_ENABLE']='1'
for abort in (False,True):
 req=NS(rid='actual-loop',req_pool_idx=1,mamba_pool_idx=1,done=False)
 req.finished=lambda:req.done
 events=[];controller=PlacementController({0:(0,1)},Adapter(),experts=4,resident=2)
 boundary=RequestBoundary(controller,events.append);calls=[]
 class SchedulerDouble:
  _engine_paused=False
  _round9_dynamic_boundary=boundary
  def recv_requests(self):
   if len(calls)>=2:raise Done()
   return []
  def process_input_requests(self,requests):
   if abort and len(calls)==1:
    req.done=True;req.req_pool_idx=req.mamba_pool_idx=None
  def get_next_batch_to_run(self):
   if abort and len(calls)==1:return None
   return NS(reqs=[req])
  def run_batch(self,batch):
   assert boundary.request is req
   boundary.observe(0,[2,2,1]);calls.append('run');return None
  def process_batch_result(self,batch,result):
   assert controller.generation==0
   if len(calls)==2:
    req.done=True;req.req_pool_idx=req.mamba_pool_idx=None
  def self_check_during_idle(self):calls.append('idle')
 try:ns['event_loop_normal'](SchedulerDouble())
 except Done:pass
 assert len(events)==1 and controller.generation==1 and boundary.request is None
 rows.append(dict(abort=abort,calls=calls,events=events))
a.output.write_text(json.dumps(dict(status='PASS_ACTUAL_F02_LOOP_HOOK_ORDER_ONLY',scheduler_sha256=hashlib.sha256(source.encode()).hexdigest(),rows=rows),indent=2)+'\n')
print('PASS_ACTUAL_F02_LOOP_HOOK_ORDER_ONLY')
