"""Stream one original layer/subset to GGUF; publish file then completion receipt.

CLI is intentionally bounded to at most4 experts without --full-layer. Does not
claim native support, perform inference, or convert all layers automatically.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import sys

if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from migration_bridge.round9.runtime.lowbit import CODEC_VERSION,SIZES,TYPES,bf16_values,encode_block,decode_block
from migration_bridge.round9.runtime.expert_source import OriginalExpertSource
from migration_bridge.round9.tools.probe_identity import verify_probe_identity


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(1024**2),b''):h.update(data)
    return h.hexdigest()


def ggstr(value):
    raw=value.encode();return struct.pack('<Q',len(raw))+raw


def convert_layer(source,layer,logical_ids,storage,output,*,source_identity,encoder="scalar"):
    if encoder not in ("scalar","numpy"):raise ValueError("unsupported encoder")
    ids=list(logical_ids);output=Path(output)
    if (not ids or ids!=sorted(set(ids)) or any(type(i) is not int or not 0<=i<source.experts for i in ids)
        or type(layer) is not int or not 0<=layer<source.layers):raise ValueError('invalid ordered expert selection')
    if storage not in SIZES or source.hidden%32 or source.intermediate%32:raise ValueError('unsupported row format/alignment')
    parent=output.parent
    while not parent.exists():parent=parent.parent
    scratch_required=2*3*len(ids)*source.hidden*source.intermediate//32*SIZES[storage]+64*1024**2
    if shutil.disk_usage(parent).free<scratch_required:raise ValueError('insufficient disk for staged and final bank plus64MiB reserve')
    output.mkdir(parents=True,exist_ok=False)
    stages=[];identities=[];error2=0.;source2=0.;max_abs=0.
    for projection in ('gate','up','down'):
        stage=output/(projection+'.partial');stages.append(stage)
        with stage.open('xb') as f:
            for expert in ids:
                raw=source.read(layer,expert,projection)
                identities.append(dict(expert=expert,projection=projection,source_bf16_sha256=hashlib.sha256(raw).hexdigest()))
                if encoder=='numpy':
                    import numpy as np
                    from migration_bridge.round9.runtime.lowbit_numpy import encode_blocks,decode_blocks
                    values=(np.frombuffer(raw,dtype='<u2').astype('<u4')<<16).view('<f4').reshape(-1,32)
                    packed=encode_blocks(values,storage);decoded=decode_blocks(packed,storage)
                    f.write(packed.tobytes())
                    delta=values.astype(np.float64)-decoded.astype(np.float64)
                    error2+=float(np.square(delta).sum());source2+=float(np.square(values.astype(np.float64)).sum())
                    max_abs=max(max_abs,float(np.abs(delta).max()))
                else:
                    for offset in range(0,len(raw),64):
                        values=bf16_values(raw[offset:offset+64]);packed=encode_block(values,storage)
                        decoded=decode_block(packed,storage)
                        f.write(packed)
                        for x,y in zip(values,decoded):
                            error2+=(x-y)**2;source2+=x*x;max_abs=max(max_abs,abs(x-y))
            f.flush();os.fsync(f.fileno())
    payload_bytes=len(ids)*source.hidden*source.intermediate//32*SIZES[storage]
    header=b'GGUF'+struct.pack('<IQQ',3,3,1)+ggstr('general.architecture')+struct.pack('<I',8)+ggstr('llama')
    offset=0;offsets=[]
    for name,stage in zip(('gate','up','down'),stages):
        if stage.stat().st_size!=payload_bytes:raise ValueError('incomplete encoded projection')
        shape=(len(ids),source.hidden,source.intermediate) if name=='down' else (len(ids),source.intermediate,source.hidden)
        header+=ggstr(f'blk.{layer}.ffn_{name}_exps.weight')+struct.pack('<I',3)+struct.pack('<3Q',*reversed(shape))+struct.pack('<IQ',TYPES[storage],offset)
        offsets.append(offset);offset=(offset+payload_bytes+31)//32*32
    header+=bytes((-len(header))%32)
    filename=f'layer-{layer:02d}-{storage.lower()}.gguf';partial=output/(filename+'.partial');final=output/filename
    with partial.open('xb') as f:
        f.write(header)
        for stage,offset in zip(stages,offsets):
            f.write(bytes(len(header)+offset-f.tell()))
            with stage.open('rb') as source_file:shutil.copyfileobj(source_file,f)
        f.flush();os.fsync(f.fileno())
    os.link(partial,final);partial.unlink()
    report=dict(status='PASS_CONVERSION_ONLY_NOT_NATIVE_QUALIFIED',codec=CODEC_VERSION,
        encoder=encoder,storage=storage,layer=layer,logical_ids=ids,scratch_required_bytes=scratch_required,complete_cpu_backing=ids==list(range(source.experts)),
        geometry=dict(experts=source.experts,hidden=source.hidden,intermediate=source.intermediate),
        file=filename,bytes=final.stat().st_size,sha256=digest(final),source_identity=source_identity,
        source_index_sha256=source.index_sha256,projection_identities=identities,
        quantization_error=dict(relative_l2=math.sqrt(error2/source2) if source2 else 0,max_abs=max_abs),
        native_status='UNSUPPORTED_NATIVE_KERNEL_160_320_GROUPED_PREFILL')
    manifest=output/'manifest.json.partial'
    with manifest.open('x') as f:json.dump(report,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    os.link(manifest,output/'manifest.json');manifest.unlink()
    for stage in stages:stage.unlink()
    return report


REVISION='de4b8e4d43b917e7706784d8bb445c9af86a3540'
TRUSTED_ROOTS=(Path('/models')/REVISION,Path('/home/admin/models/Qwen3.8-Flash-Next')/REVISION)


def require_trusted_root(model, *, readonly):
    root=Path(model).resolve()
    if root not in tuple(p.resolve() for p in TRUSTED_ROOTS):
        raise ValueError('source is not an approved original model root')
    if readonly is not True:raise ValueError('conversion requires read-only original model mount')
    return root


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True)
    p.add_argument('--layer',type=int,choices=range(48),required=True);p.add_argument('--experts',default='0,1,510,511')
    p.add_argument('--full-layer',action='store_true');p.add_argument('--storage',choices=SIZES,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--encoder',choices=['scalar','numpy'],default='scalar')
    a=p.parse_args();revision='de4b8e4d43b917e7706784d8bb445c9af86a3540'
    trusted=require_trusted_root(a.model,readonly=bool(os.statvfs(a.model).f_flag & os.ST_RDONLY))
    identity=verify_probe_identity(model=trusted,receipt=a.receipt,expected_model=trusted,
        expected_receipt_sha256='7d6cb3db19497303bdd8ceb24efae626e1926a610ab7fbd4406b47d05db4a239',revision=revision,
        manifest_sha256='62e6eca0f2900745dbfd9ba6ed76e60547e0d602acf2d62a9ecac3972028b38e')
    ids=list(range(512)) if a.full_layer else [int(x) for x in a.experts.split(',')]
    if not a.full_layer and len(ids)>4:raise ValueError('small sample limit4; full-layer requires explicit flag')
    print(json.dumps(convert_layer(OriginalExpertSource(a.model),a.layer,ids,a.storage,a.output,source_identity=identity,encoder=a.encoder),indent=2))

if __name__=='__main__':main()
