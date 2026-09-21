"""Bounded synchronous pread of original PLE BF16 rows; no full table copy.

Requires the model's separate source identity gate. This reader validates table
headers/offsets; it does not replace full checkpoint content verification.
"""
import bisect
import json
import os
import pathlib
import re
import struct

PREFIX='model.language_model.layers.1.ple.ple_embedding.ngram_embedding.'
class ShardedBF16Rows:
    def __init__(self,model,*,expected_parts=128,expected_width=160):
        self.root=pathlib.Path(model).resolve(strict=True)
        index=json.loads((self.root/'model.safetensors.index.json').read_text())['weight_map']
        names=[k for k in index if k.startswith(PREFIX)]
        expected=[f'{PREFIX}shard_{i}.weight' for i in range(expected_parts)]
        if set(names)!=set(expected):raise ValueError('missing or unexpected original PLE shards')
        self.width=expected_width;self.row_bytes=2*expected_width
        self.parts=[];self.ends=[];self._fds={};self.closed=False
        total=0
        for name in expected:
            path=(self.root/index[name]).resolve(strict=True)
            if not path.is_relative_to(self.root):raise ValueError('checkpoint path escape')
            with path.open('rb') as f:
                raw=f.read(8)
                if len(raw)!=8:raise ValueError('truncated header')
                n=struct.unpack('<Q',raw)[0]
                if not 0<n<=16*1024*1024:raise ValueError('invalid header length')
                header=json.loads(f.read(n));item=header[name]
            shape=item['shape'];begin,end=item['data_offsets']
            if (item['dtype']!='BF16' or len(shape)!=2 or shape[1]!=expected_width
                    or type(shape[0]) is not int or shape[0]<=0
                    or type(begin) is not int or type(end) is not int
                    or begin<0 or end-begin!=shape[0]*self.row_bytes
                    or 8+n+end>path.stat().st_size):
                raise ValueError('invalid original PLE tensor layout')
            self.parts.append((path,8+n+begin,total));total+=shape[0];self.ends.append(total)
        self.total_rows=total

    def gather_bytes(self,ids):
        if self.closed:raise RuntimeError('table reader closed')
        ids=list(ids)
        if any(type(i) is not int or not 0<=i<self.total_rows for i in ids):
            raise ValueError('PLE row outside original table')
        cache={};out=[]
        for row in ids:
            if row not in cache:
                part=bisect.bisect_right(self.ends,row)
                path,start,base=self.parts[part]
                if path not in self._fds:self._fds[path]=os.open(path,os.O_RDONLY)
                data=os.pread(self._fds[path],self.row_bytes,start+(row-base)*self.row_bytes)
                if len(data)!=self.row_bytes:raise ValueError('short original row read')
                cache[row]=data
            out.append(cache[row])
        return b''.join(out)

    def close(self):
        for fd in self._fds.values():os.close(fd)
        self._fds.clear();self.closed=True
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
