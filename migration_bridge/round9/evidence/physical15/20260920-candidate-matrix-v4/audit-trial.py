import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]);output=root/'post-trial-audit.json'
if output.exists():raise FileExistsError(output)
summary=json.loads((root/'trial-summary.json').read_text())
report={'scope':'EVIDENCE_HASHES_AND_DYNAMIC_TRANSACTION_METADATA_NOT_NUMERIC_ORACLE','status':'RUNNING'}
try:
 assert summary.get('resource_system_status')=='PASS_CANDIDATE_RESOURCE_SYSTEM' or summary['status']=='PASS_CANDIDATE_SYSTEM_RESOURCE_ONLY'
 report['candidate_overall_status']=summary['status']
 for name,expected in summary['evidence_sha256'].items():
  assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected,name
 report['verified_files']=len(summary['evidence_sha256'])
 event_path=root/'dynamic-events.jsonl'
 if event_path.exists():
  events=[json.loads(s) for s in event_path.read_text().splitlines()]
  assert events[0]['layers']==48 and events[0]['generation']==0
  generation=0;states={};ids=set();copied=0
  for event in events[1:]:
   rid=event['request_id'];assert rid not in ids;ids.add(rid)
   plans=event['plans'];assert event['changed_layers']==len(plans)
   assert len({p['layer'] for p in plans})==len(plans)
   assert event['copied_experts']==sum(len(p['moves']) for p in plans)
   generation+=bool(plans);assert event['generation']==generation
   copied+=event['copied_experts']
   for p in plans:
    layer=p['layer'];assert type(layer) is int and 0<=layer<48
    old=p['previous'];new=p['slots']
    for slots in (old,new):assert len(slots)==64 and len(set(slots))==64 and all(type(i) is int and 0<=i<512 for i in slots)
    assert layer not in states or states[layer]==old
    assert p['moves']==[[i,e] for i,e in enumerate(new) if e!=old[i]] and p['moves']
    states[layer]=new
  required=[]
  for name in ('first-http','context4096'):
   required.append(json.loads((root/(name+'.json')).read_text())['response']['meta_info']['id'])
  for name in ('system8','canaries'):
   required.extend(row['response']['meta_info']['id'] for row in json.loads((root/(name+'.json')).read_text())['rows'])
  required+=['r9-cancel-active','r9-post-cancel-reuse']
  assert set(required)<=ids,sorted(set(required)-ids)
  report.update(dynamic_transactions=len(events)-1,final_generation=generation,copied_experts=copied,required_requests_drained=len(set(required)),dynamic_log_sha256=hashlib.sha256(event_path.read_bytes()).hexdigest())
 report['status']='PASS_EVIDENCE_AND_BOUNDARY_AUDIT'
except BaseException as exc:
 report.update(status='FAIL',error=repr(exc));raise
finally:output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
