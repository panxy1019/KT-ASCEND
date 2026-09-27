"""Validate complete observed Layerwise batches; host totals alone are insufficient."""
def validate(document, *, expected_request, expected_input_tokens, cancelled=False):
 if document['status']!='COMPLETE_HOST_ACCOUNTING':raise ValueError('incomplete host ledger')
 identity=document['identity']
 if identity['request_id']!=expected_request or identity['input_tokens']!=expected_input_tokens:raise ValueError('request identity mismatch')
 if type(identity.get('request_allocation_epoch')) is not int:raise ValueError('missing request allocation epoch')
 if document.get('termination') != ('cancelled' if cancelled else 'finished'):raise ValueError('termination mismatch')
 events=document['events'];byid={r['id']:r for r in events}
 recomputed={}
 for row in events:
  entry=recomputed.setdefault(row['stage'],dict(calls=0,inclusive_ns=0,self_ns=0,logical_bytes=0,expert_count=0))
  entry['calls']+=1
  for key in ('inclusive_ns','self_ns','logical_bytes','expert_count'):entry[key]+=row[key]
 if document.get('totals')!=recomputed:raise ValueError('totals disagree with events')
 if len(byid)!=len(events):raise ValueError('duplicate span IDs')
 def descendants(root):
  found=[]
  for row in events:
   parent=row['parent'];seen=set()
   while parent is not None:
    if parent in seen or parent not in byid:raise ValueError('invalid parent graph')
    seen.add(parent)
    if parent==root['id']:found.append(row);break
    parent=byid[parent]['parent']
  return found
 batches=sorted((r for r in events if r['stage']=='prefill_batch'),key=lambda r:r['start_ns'])
 position=0;group_count=0;logical_bytes=0;seen_groups=[]
 for batch in (r for r in events if r['stage'] in ('prefill_batch','decode_batch')):
  children=descendants(batch)
  for stage in ('model_forward','result_processing'):
   if sum(r['stage']==stage for r in children)!=1:raise ValueError('model/result seam coverage')
 for batch in batches:
  start,count=batch['chunk']
  if start!=position or not 1<=count<=32:raise ValueError('chunk gap/bound')
  position+=count
  rows=descendants(batch)
  layers=[r for r in rows if r['stage']=='expert_layer']
  if sorted(r['layer'] for r in layers)!=list(range(48)):raise ValueError('missing/duplicate layer')
  for layer in sorted(layers,key=lambda r:r['start_ns']):
   children=descendants(layer)
   for stage,expected in [('route_collect',2),('route_group',1),('assemble',1)]:
    if sum(r['stage']==stage for r in children)!=expected:raise ValueError('layer stage coverage: '+stage)
   groups=[r for r in children if r['stage']=='expert_group']
   if not groups or sum(r['token_rows'] for r in groups)!=count:raise ValueError('group token coverage')
   gids={r['group'] for r in groups}
   if len(gids)!=len(groups):raise ValueError('duplicate group identity')
   for stage in ('prepare','compute_enqueue','device_wait','release'):
    stage_rows=[r for r in children if r['stage']==stage]
    if len(stage_rows)!=len(groups) or {r['group'] for r in stage_rows}!=gids:raise ValueError('group stage coverage: '+stage)
    if any(r['layer']!=layer['layer'] or r['chunk']!=batch['chunk'] for r in stage_rows):raise ValueError('group layer/chunk identity')
   for group in sorted(groups,key=lambda r:r['start_ns']):
    if group['layer']!=layer['layer'] or group['chunk']!=batch['chunk']:raise ValueError('group metadata mismatch')
    if type(group['group']) is not int or (seen_groups and group['group']<=seen_groups[-1]):raise ValueError('reused/nonmonotonic group identity')
    seen_groups.append(group['group'])
    if not 10<=group['expert_count']<=32:raise ValueError('expert count outside Q32 Top10')
    prepare=next(r for r in children if r['stage']=='prepare' and r['group']==group['group'])
    if prepare['expert_count']!=group['expert_count'] or prepare['logical_bytes']!=group['expert_count']*2560*640*3*2:raise ValueError('logical weight bytes mismatch')
    logical_bytes+=prepare['logical_bytes']
   group_count+=len(groups)
 if position>expected_input_tokens or (not cancelled and position!=expected_input_tokens):raise ValueError('incomplete/excess prefill')
 for batch in (r for r in events if r['stage']=='decode_batch'):
  if any(r['stage'] in ('expert_layer','prepare','expert_group') for r in descendants(batch)):raise ValueError('Layerwise transport on decode')
 if not batches:raise ValueError('no prefill coverage')
 return dict(status='PASS_48_LAYER_GROUP_COVERAGE',prefill_batches=len(batches),processed_input_tokens=position,groups=group_count,prepare_calls=group_count,logical_weight_bytes=logical_bytes,device_bytes='UNKNOWN')
