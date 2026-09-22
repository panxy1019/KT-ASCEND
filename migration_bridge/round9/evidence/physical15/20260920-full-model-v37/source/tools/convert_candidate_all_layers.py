"""Bounded offline48-layer conversion, gated by a real full-bank KTEP receipt.

Creates a new catalog only after every layer finishes. No inference or timer.
"""
import argparse,concurrent.futures,hashlib,json,os,pathlib,shutil,sys,time
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[3]))
from migration_bridge.round9.tools.convert_candidate_bank import convert_layer,require_trusted_root,digest
from migration_bridge.round9.runtime.expert_source import OriginalExpertSource
from migration_bridge.round9.tools.probe_identity import verify_probe_identity

def worker(job):
    model,layer,storage,folder,identity=job
    report=convert_layer(OriginalExpertSource(model),layer,list(range(512)),storage,folder,source_identity=identity,encoder='numpy')
    return layer,str(folder),report['sha256']

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model','receipt','gate','layer0','output','catalog'):p.add_argument('--'+name,type=pathlib.Path,required=True)
    p.add_argument('--storage',choices=['Q8_0','MXFP4'],required=True)
    a=p.parse_args();rev='de4b8e4d43b917e7706784d8bb445c9af86a3540'
    root=require_trusted_root(a.model,readonly=bool(os.statvfs(a.model).f_flag&os.ST_RDONLY))
    identity=verify_probe_identity(model=root,receipt=a.receipt,expected_model=root,expected_receipt_sha256='7d6cb3db19497303bdd8ceb24efae626e1926a610ab7fbd4406b47d05db4a239',revision=rev,manifest_sha256='62e6eca0f2900745dbfd9ba6ed76e60547e0d602acf2d62a9ecac3972028b38e')
    gate=json.loads(a.gate.read_text());first=json.loads((a.layer0/'manifest.json').read_text())
    if not (gate['status']=='PASS_REAL_KTEP_LOCAL_MERGE' and gate['storage']==a.storage):raise RuntimeError('conversion admission or bank identity check failed')
    if not (gate['source_identity']==identity and gate['bank_sha256']==first['sha256']):raise RuntimeError('conversion admission or bank identity check failed')
    if not (first['layer']==0 and first['storage']==a.storage and first['complete_cpu_backing'] and first['logical_ids']==list(range(512))):raise RuntimeError('conversion admission or bank identity check failed')
    if not (digest(a.layer0/first['file'])==first['sha256']):raise RuntimeError('conversion admission or bank identity check failed')
    required=48*first['bytes']+8*first['bytes']+64*1024**3
    if not (shutil.disk_usage(a.output.parent).free>required):raise RuntimeError('insufficient final+scratch+reserve disk')
    if not (not a.catalog.exists()):raise RuntimeError('conversion admission or bank identity check failed')
    a.output.mkdir(exist_ok=False)
    (a.output/'conversion-admission.json').write_text(json.dumps(dict(time=time.time(),status='OFFLINE_CONVERSION_ONLY',source_identity=identity,gate_sha256=digest(a.gate),storage=a.storage,workers=4,required_free_bytes=required),indent=2)+'\n')
    jobs=[(str(root),layer,a.storage,a.output/f'layer-{layer:02d}',identity) for layer in range(1,48)]
    pool=concurrent.futures.ProcessPoolExecutor(max_workers=4)
    try:
        futures=[pool.submit(worker,job) for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            layer,folder,sha=future.result();print(json.dumps(dict(layer=layer,status='CONVERTED',sha256=sha,time=time.time())),flush=True)
    except BaseException:
        pool.shutdown(wait=True,cancel_futures=True);raise
    else:pool.shutdown()
    rows=[]
    for layer in range(48):
        folder=a.layer0 if layer==0 else a.output/f'layer-{layer:02d}'
        item=json.loads((folder/'manifest.json').read_text())
        if not (item['layer']==layer and item['storage']==a.storage and item['source_identity']==identity):raise RuntimeError('conversion admission or bank identity check failed')
        if not (item['complete_cpu_backing'] and item['logical_ids']==list(range(512))):raise RuntimeError('conversion admission or bank identity check failed')
        if not (item['geometry']==dict(experts=512,hidden=2560,intermediate=640)):raise RuntimeError('conversion admission or bank identity check failed')
        if not (item['status']=='PASS_CONVERSION_ONLY_NOT_NATIVE_QUALIFIED'):raise RuntimeError('conversion admission or bank identity check failed')
        if not (len(item['projection_identities'])==1536 and (folder/item['file']).stat().st_size==item['bytes']):raise RuntimeError('conversion admission or bank identity check failed')
        if digest(folder/item['file'])!=item['sha256']:raise RuntimeError('bank SHA changed before publication')
        rows.append(dict(item,manifest_sha256=digest(folder/'manifest.json'),bank_path=str((folder/item['file']).resolve())))
    a.catalog.mkdir()
    for item in rows:(a.catalog/item['file']).symlink_to(os.path.relpath(item['bank_path'],a.catalog))
    report=dict(conversion_admission_sha256=digest(a.output/'conversion-admission.json'),gate_sha256=digest(a.gate),converter_sha256=digest(pathlib.Path(__file__)),status='PASS_48_CONVERTED_BANK_RECEIPTS_AND_LIVE_SIZES_NOT_MODEL_QUALIFIED',storage=a.storage,source_identity=identity,banks=rows,total_bytes=sum(r['bytes'] for r in rows))
    (a.catalog/'catalog.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report['status'],flush=True)
if __name__=='__main__':main()
