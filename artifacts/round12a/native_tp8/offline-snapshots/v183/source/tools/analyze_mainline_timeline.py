"""Conservative host interval accounting; never infers device overlap."""
import argparse
import json
import math
from pathlib import Path


def interval_union(intervals):
    end=None;total=0
    for lo,hi in sorted(intervals):
        if hi<lo:raise ValueError('reversed interval')
        total+=max(0,hi-max(lo,end if end is not None else lo))
        end=max(hi,end if end is not None else hi)
    return total


def summarize(documents):
    intervals=[];components={};invalid=0;dropped=0;decode=0;requests=[];missing_coverage=0;observer_errors=[];coverage=0
    for doc in documents:
        dropped+=doc.get('dropped_events',0)
        requests.append(doc.get('identity'))
        local_coverage=0
        for event in doc.get('events',[]):
            if event.get('category')!='mainline':continue
            if event.get('phase')=='batch_coverage':
                dropped+=event.get('dropped_spans',0);coverage+=1;local_coverage+=1
                observer_errors.extend(event.get('observer_errors',[]));continue
            if event.get('phase')!='span':continue
            try:
                lo,hi=event['start_ns'],event['end_ns']
                if type(lo) is not int or type(hi) is not int or hi<lo:raise ValueError()
                if not isinstance(event.get('component'),str) or not event['component']:raise ValueError()
                ms=event.get('device_ms')
                if ms is not None and (type(ms) not in (float,int) or not math.isfinite(ms) or ms<0):raise ValueError()
            except (KeyError,ValueError):invalid+=1;continue
            observer_errors.extend(event.get('observer_errors',[]))
            intervals.append((lo,hi));decode+=event.get('mode')=='decode'
            key=event['component']+':'+event.get('mode','UNKNOWN')
            row=components.setdefault(key,dict(calls=0,inclusive_host_ns=0,device_duration_ms=[],device_missing=0))
            row['calls']+=1;row['inclusive_host_ns']+=hi-lo
            if event.get('device_ms') is None:row['device_missing']+=1
            else:row['device_duration_ms'].append(event['device_ms'])
        missing_coverage+=local_coverage==0
    device_samples=sum(len(row['device_duration_ms']) for row in components.values())
    status=('EVIDENCE_INCOMPLETE' if not intervals or missing_coverage or observer_errors else
            'PARTIAL_TRUNCATED' if invalid or dropped else
            'PARTIAL_HOST_ONLY' if not device_samples else 'PARTIAL_ATTRIBUTION')
    return dict(schema=1,status=status,coverage_batches=coverage,missing_request_coverage=missing_coverage,
                observer_errors=observer_errors,device_duration_samples=device_samples,
                host_union_ns=interval_union(intervals) if intervals else None,
                components=components,invalid_spans=invalid,dropped_events=dropped,
                decode_spans=decode,cpu_worker_ns=None,dram_bandwidth=None,
                device_overlap='UNKNOWN_UNCALIBRATED_CLOCKS',requests=requests,
                interpretation='Inclusive sampled durations; no mutually exclusive percentages or full critical path.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('input',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    a.output.write_text(json.dumps(summarize([json.loads(l) for l in a.input.read_text().splitlines()]),indent=2)+'\n')
