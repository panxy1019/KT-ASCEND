"""Guarded original-table PLE transport probe; no model or performance claim."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import sys
import threading
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('model','identity-receipt','admission','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    base=Path(__file__).resolve().parents[1]
    sys.path[:0]=[str(base),str(base/'tools')]
    from probe_identity import verify_probe_identity
    from runtime.ple_gate import SOURCE_FILES,CHECKS,require_ple_device_gate
    from runtime import ple_async,ple_rows,host_budget,framework_config
    admission=json.loads(args.admission.read_text())
    source_sha={name:hashlib.sha256((base/'runtime'/name).read_bytes()).hexdigest() for name in SOURCE_FILES}
    if (admission.get('status')!='ADMITTED_PLE_COMPONENT_PENDING' or
            admission.get('physical')!=15 or admission.get('source_sha256')!=source_sha or
            admission.get('probe_sha256')!=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()):
        raise RuntimeError('missing source-bound exclusive PLE component admission')
    if args.output.exists():raise FileExistsError(args.output)
    rev='de4b8e4d43b917e7706784d8bb445c9af86a3540'
    identity=verify_probe_identity(model=args.model,receipt=args.identity_receipt,
        expected_model=Path('/models')/rev,revision=rev,
        expected_receipt_sha256='7d6cb3db19497303bdd8ceb24efae626e1926a610ab7fbd4406b47d05db4a239',
        manifest_sha256='62e6eca0f2900745dbfd9ba6ed76e60547e0d602acf2d62a9ecac3972028b38e')
    import torch
    import torch_npu
    torch.set_num_threads(2)
    torch.npu.set_device(0)
    torch.npu.set_per_process_memory_fraction(.01,0)
    physical=ctypes.c_int32(-1)
    acl=ctypes.CDLL('/usr/local/Ascend/cann-9.0.0/aarch64-linux/lib64/libascendcl.so')
    acl.aclrtGetPhyDevIdByLogicDevId.argtypes=[ctypes.c_int32,ctypes.POINTER(ctypes.c_int32)]
    if (acl.aclrtGetPhyDevIdByLogicDevId(0,ctypes.byref(physical))!=0 or physical.value!=15 or
            torch.npu.device_count()!=1 or set(Path('/dev').glob('davinci[0-9]*'))!={Path('/dev/davinci15')}):
        raise RuntimeError('PLE probe requires isolated physical15')
    torch.npu.synchronize();torch.npu.reset_peak_memory_stats()
    baseline=int(torch.npu.memory_allocated())
    budget=host_budget.HostWorkBudget(enabled=True,host_bytes=16*1024**2,
        h2d_bytes=16*1024**2,pinned_bytes=327680)
    reader=ple_rows.ShardedBF16Rows(args.model)
    block=threading.Event();started=threading.Event()
    class ControlledReader:
        width=reader.width
        total_rows=reader.total_rows
        stall=False
        def gather_bytes(self,ids):
            if self.stall:
                started.set()
                if not block.wait(10):raise TimeoutError('bounded cancelled gather timed out')
            return reader.gather_bytes(ids)
    controlled=ControlledReader()
    ring=ple_async.AsyncPLERing(controlled,enabled=True,max_rows=512,budget=budget)
    checks={key:False for key in CHECKS};cases=[]
    flat=[0,reader.total_rows-1]
    for end in reader.ends[:-1]:flat.extend((end-1,end))
    flat=(flat+flat)[:512]
    assert len(flat)==512
    rows=[flat[i:i+16] for i in range(0,len(flat),16)]
    def case(rid,epoch,ids,phase):
        expected=reader.gather_bytes([i for row in ids for i in row])
        ticket=ring.submit_exact(rid,epoch,ids,phase=phase)
        assert ticket is not None
        consumer=torch.npu.Stream()
        with torch.npu.stream(consumer):
            value=ring.wait(ticket,consumer_stream=consumer)
            copied=value.clone()
        ring.release(ticket)  # records/drains consumer event, not just H2D event
        actual=copied.cpu().contiguous().view(torch.uint8).numpy().tobytes()
        assert actual==expected
        cases.append(dict(request_id=rid,generation=epoch,rows=len(ids)*len(ids[0]),
                          sha256=hashlib.sha256(actual).hexdigest(),exact=True))
        budget.retire_request(rid)
        return ticket
    first=case('prefill-original',0,rows,ple_async.PREFILL_INPUT)
    checks.update(original_rows_exact=True,duplicates_exact=True,cross_shard_exact=True,
                  consumer_stream_order=True)
    second=case('sampled-next',1,[rows[1]],ple_async.SAMPLED_DECODE)
    assert first.slot==second.slot and first.identity!=second.identity
    try:ring.poll(first)
    except ValueError:pass
    else:raise AssertionError('retired ticket remained usable')
    checks.update(slot_reuse=True,request_switch=True)
    controlled.stall=True
    cancelled=ring.submit_exact('abort-pending',1,[rows[2]],phase=ple_async.PREFILL_INPUT)
    assert started.wait(5)
    before=budget.snapshot()['used'].copy()
    ring.cancel(cancelled)
    assert budget.snapshot()['used']==before
    block.set()
    deadline=time.monotonic()+10
    while ring.poll(cancelled)!='RETIRED':
        if time.monotonic()>deadline:raise TimeoutError('cancel retirement timed out')
        time.sleep(.01)
    budget.retire_request('abort-pending');controlled.stall=False
    case('reuse-after-abort',2,[rows[3]],ple_async.SAMPLED_DECODE)
    checks['cancel_late_completion']=True
    torch.npu.synchronize()
    peak=int(torch.npu.max_memory_allocated())-baseline
    ring.close();reader.close()
    assert budget.snapshot()['allocations']==0 and not any(budget.snapshot()['used'].values())
    checks['zero_credits_after_close']=True
    gate=dict(schema=1,status='PASS_PLE_ASYNC_ORIGINAL_ROWS',
        scope='COMPONENT_ONLY_NOT_MODEL_OR_OVERLAP_PERFORMANCE',physical_device=15,
        source_identity=identity,source_sha256=source_sha,torch_version=str(torch.__version__),
        torch_npu_version=str(torch_npu.__version__),slots=2,max_rows=512,row_width=160,
        dtype='torch.bfloat16',pinned_bytes=327680,device_buffer_bytes=327680,
        peak_extra_bytes=peak,checks=checks,cases=cases)
    raw=(json.dumps(gate,sort_keys=True,indent=2)+'\n').encode()
    temporary=args.output.with_suffix('.pending.json');temporary.write_bytes(raw)
    require_ple_device_gate(temporary,hashlib.sha256(raw).hexdigest(),source_identity=identity,
        source_sha256=source_sha,torch_version=str(torch.__version__),torch_npu_version=str(torch_npu.__version__))
    temporary.replace(args.output)
    print(gate['status'],flush=True)


if __name__=='__main__':main()
