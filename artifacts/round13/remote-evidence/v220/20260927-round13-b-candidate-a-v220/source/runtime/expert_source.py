"""Bounded original BF16 expert reads, never low-bit dequantization for promotion.

A separately verified immutable checkpoint receipt is required by operational
callers. Header validation and per-projection SHA do not replace that receipt.
"""
import hashlib
import json
import math
from pathlib import Path
import struct


class OriginalExpertSource:
    def __init__(self, model, *, layers=48, experts=512, hidden=2560, intermediate=640):
        self.root=Path(model).resolve(strict=True)
        self.layers=layers;self.experts=experts;self.hidden=hidden;self.intermediate=intermediate
        if any(type(x) is not int or x<=0 for x in (layers,experts,hidden,intermediate)):
            raise ValueError('invalid expert geometry')
        raw=(self.root/'model.safetensors.index.json').read_bytes()
        self.index=json.loads(raw)['weight_map'];self.index_sha256=hashlib.sha256(raw).hexdigest()
        self._headers={}

    def read(self, layer, expert, projection):
        if type(layer) is not int or not 0<=layer<self.layers or type(expert) is not int or not 0<=expert<self.experts:
            raise ValueError('invalid logical layer/expert')
        if projection not in ('gate','up','down'):raise ValueError('invalid projection')
        key=f'model.language_model.layers.{layer}.mlp.experts.'+('down_proj' if projection=='down' else 'gate_up_proj')
        path=(self.root/self.index[key]).resolve(strict=True)
        if not path.is_relative_to(self.root):raise ValueError('checkpoint path escape')
        stat=path.stat();signature=(stat.st_size,stat.st_mtime_ns,stat.st_ino)
        if path not in self._headers:
            with path.open('rb') as f:
                first=f.read(8)
                if len(first)!=8:raise ValueError('truncated header')
                n=struct.unpack('<Q',first)[0]
                if not 0<n<=16*1024**2:raise ValueError('invalid safetensors header')
                data=f.read(n)
                if len(data)!=n:raise ValueError('truncated header')
                self._headers[path]=(json.loads(data),n+8,signature)
        header,base,original=self._headers[path]
        if signature!=original:raise ValueError('original source changed while in use')
        shape=([self.experts,self.hidden,self.intermediate] if projection=='down' else
               [self.experts,2*self.intermediate,self.hidden])
        info=header[key];begin,end=info['data_offsets']
        if (info['dtype']!='BF16' or info['shape']!=shape or type(begin) is not int or type(end) is not int or begin<0 or end-begin!=math.prod(shape)*2 or base+end>stat.st_size):
            raise ValueError('invalid original BF16 expert layout')
        count=self.hidden*self.intermediate*2
        offset=(expert*count if projection=='down' else expert*2*count+(count if projection=='up' else 0))
        with path.open('rb') as f:
            f.seek(base+begin+offset);raw=f.read(count)
        if len(raw)!=count:raise ValueError('short original expert read')
        return raw
