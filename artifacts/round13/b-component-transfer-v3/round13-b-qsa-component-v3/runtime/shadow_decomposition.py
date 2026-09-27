"""Bounded, opt-in shadow preparation observation. No compute or synchronization."""
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
import math
import time

_active = ContextVar('round9_shadow_observer', default=None)
_off = nullcontext()

def phase(name, *, device=False, **metadata):
    recorder = _active.get()
    return _off if recorder is None else recorder.span(name, device, metadata)

@contextmanager
def activate(recorder):
    token = _active.set(recorder)
    try: yield
    finally: _active.reset(token)

class Recorder:
    def __init__(self, *, max_events=2048, device_factory=None, run_id="UNSPECIFIED"):
        if not 1 <= max_events <= 2048: raise ValueError('event budget')
        self.max_events=max_events;self.device_factory=device_factory;self.run_id=run_id;self.context={}
        self.events=[];self.stack=[];self.open_rows={};self.errors=[];self.dropped=0;self.sequence=0
        self.pending=[];self.device_broken=False
    def error(self, exc):
        self.device_broken=True
        if len(self.errors)<8:self.errors.append(type(exc).__name__+': '+str(exc)[:256])
    @contextmanager
    def span(self, name, device, metadata):
        if self.sequence>=self.max_events:
            self.dropped+=1
            yield
            return
        self.sequence+=1
        row=dict(id=self.sequence,parent=self.stack[-1] if self.stack else None,
                 name=name,metadata=metadata,clock='HOST_MONOTONIC_NS',device_ms=None,run_id=self.run_id,context=dict(self.context))
        pair=None
        if device and self.device_factory and not self.device_broken:
            try:pair=self.device_factory();pair[0].record()
            except Exception as exc:self.error(exc);pair=None
        row['start_ns']=time.monotonic_ns();self.stack.append(row['id']);self.open_rows[row['id']]=row
        row['outcome']='error'
        try:
            yield
            row['outcome']='return'
        finally:
            row['end_ns']=time.monotonic_ns();self.stack.pop();self.open_rows.pop(row['id'])
            if pair is not None:
                try:pair[1].record();self.pending.append((row,pair))
                except Exception as exc:self.error(exc)
            self.events.append(row)
    def drain_events(self):
        """Call only after the provider's existing drain. Does not synchronize."""
        for row,(start,end) in self.pending:
            try:
                if not start.query() or not end.query():raise RuntimeError('event incomplete at existing drain')
                value=float(start.elapsed_time(end))
                if not math.isfinite(value) or value<0:raise ValueError('invalid device duration')
                row['device_ms']=value
            except Exception as exc:self.error(exc)
        self.pending.clear()
    def report(self):
        result=summarize(self.events)
        result.update(errors=self.errors,dropped=self.dropped,pending_events=len(self.pending))
        if self.errors or self.dropped or self.pending:result['status']='INCOMPLETE'
        return result

def summarize(events):
    """Immediate-child union only; cross-parent intervals are invalid, never clipped."""
    if not events:return dict(status='INCOMPLETE',parents=[],invalid=['empty'])
    by_id={};invalid=[]
    for row in events:
        if (type(row.get('id')) is not int or row['id'] in by_id or
            type(row.get('start_ns')) is not int or type(row.get('end_ns')) is not int or
            row['end_ns']<row['start_ns']):invalid.append('malformed/duplicate');continue
        by_id[row['id']]=row
    for row in by_id.values():
        if row.get('parent') is not None:
            parent=by_id.get(row['parent'])
            if parent is None or not parent['start_ns']<=row['start_ns']<=row['end_ns']<=parent['end_ns']:
                invalid.append('missing parent or child outside parent')
    parents=[];phases={}
    errored=any(x.get('outcome','return')!='return' for x in events)
    if not invalid:
        for row in by_id.values():
            if row['name']!='prepare':continue
            spans=sorted((x['start_ns'],x['end_ns']) for x in by_id.values() if x.get('parent')==row['id'])
            covered=0;end=row['start_ns']
            for lo,hi in spans:covered+=max(0,hi-max(lo,end));end=max(end,hi)
            duration=row['end_ns']-row['start_ns']
            parents.append(dict(id=row['id'],metadata=row.get('metadata',{}),duration_ns=duration,covered_ns=covered,residual_ns=duration-covered))
    if not invalid:
        for row in by_id.values():
            entry=phases.setdefault(row['name'],dict(calls=0,inclusive_host_ns=0,intervals=[],device_ms=[]))
            entry['calls']+=1;entry['inclusive_host_ns']+=row['end_ns']-row['start_ns'];entry['intervals'].append((row['start_ns'],row['end_ns']))
            ms=row.get('device_ms')
            if ms is not None:
                if type(ms) not in (float,int) or not math.isfinite(ms) or ms<0:invalid.append('invalid device duration')
                else:entry['device_ms'].append(ms)
        for entry in phases.values():
            end=None;covered=0
            for lo,hi in sorted(entry.pop('intervals')):
                covered+=hi-lo if end is None else max(0,hi-max(lo,end));end=hi if end is None else max(end,hi)
            entry['host_union_ns']=covered
            if not entry['device_ms']:entry['device_ms']=None
    return dict(status='INVALID' if invalid else 'INCOMPLETE' if errored else 'VALID_HOST_DECOMPOSITION' if parents else 'INCOMPLETE',parents=parents,phases=phases,invalid=invalid,device_overlap='UNKNOWN_UNCALIBRATED',dram_bandwidth=None,interpretation='Phase rows are inclusive/nested, never sum phases as exclusive costs; device durations independent.')

def record_tensor(name, tensor, *, layout):
    recorder=_active.get()
    if recorder is None or not recorder.stack:return
    try:
        storage=tensor.untyped_storage()
        identity=dict(shape=list(tensor.shape),stride=list(tensor.stride()),dtype=str(tensor.dtype),device=str(tensor.device),pointer=tensor.data_ptr(),storage_pointer=storage.data_ptr(),storage_bytes=storage.nbytes(),storage_offset=tensor.storage_offset(),layout=layout)
        tensors=recorder.open_rows[recorder.stack[-1]]['metadata'].setdefault('tensors',{})
        if len(tensors)>=4:raise ValueError('tensor metadata budget')
        tensors[name]=identity
    except Exception as exc:recorder.error(exc)
