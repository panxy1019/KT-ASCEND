"""Audit contiguous frozen reference slices without inheriting another score."""
import argparse,hashlib,json,math,pathlib
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=pathlib.Path,required=True);p.add_argument('--slices',type=pathlib.Path,nargs='+',required=True);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
manifest=json.loads(a.manifest.read_text());digest=hashlib.sha256(a.manifest.read_bytes()).hexdigest()
assert manifest['revision']=='de4b8e4d43b917e7706784d8bb445c9af86a3540' and len(manifest['rows'])==256
records=[];ranges=[];next_start=0;driver=None;runtime=None
for root in a.slices:
 summary=json.loads((root/'summary.json').read_text());assert summary['status']=='REFERENCE_SLICE_SCORED_PENDING_AUDIT'
 assert summary['manifest_sha256']==digest and summary['invalid']==0 and summary['start']==next_start
 assert hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()==digest
 assert summary['freeze_ack']['manifest_sha256']==digest
 driver=driver or summary['driver_sha256'];assert summary['driver_sha256']==driver
 runtime=runtime or summary['runtime_identity'];assert summary['runtime_identity']==runtime and runtime['observer'] is False
 rows=[json.loads(l) for l in (root/'raw.jsonl').read_text().splitlines()]
 assert len(rows)==summary['count'];records.extend(rows);ranges.append([next_start,next_start+len(rows)]);next_start+=len(rows)
assert next_start==256 and len({r['id'] for r in records})==256
correct=0;predictions=[]
for frozen,raw in zip(manifest['rows'],records):
 assert raw['id']==frozen['id'] and raw['request']['input_ids']==frozen['input_ids']
 assert raw['request']['token_ids_logprob']==frozen['candidate_ids']
 assert raw['request']['sampling_params']==dict(temperature=0,max_new_tokens=1,stop=frozen['stop'])
 meta=raw['response']['meta_info'];assert meta['id']==raw['request']['rid'] and meta['completion_tokens']==1
 positions=meta['output_token_ids_logprobs'];assert len(positions)==1
 scores={v[1]:v[0] for v in positions[0]};assert set(scores)==set(frozen['candidate_ids']) and all(math.isfinite(v) for v in scores.values())
 prediction='ABCD'[max(range(4),key=lambda j:scores[frozen['candidate_ids'][j]])]
 predictions.append(prediction);correct+=int(prediction==frozen['answer'])
result=dict(status='PASS_FROZEN256_REFERENCE_AUDIT',scope='FRESH_Q64_BF16_STORAGE_F32_CPU_C64_ENGINEERING_REFERENCE_NOT_ALL_NPU',manifest_sha256=digest,driver_sha256=driver,runtime_identity=runtime,count=256,correct=correct,accuracy=correct/256,invalid=0,ordered_ranges=ranges,answer_distribution={c:predictions.count(c) for c in 'ABCD'},inherited_model_scores=False,formal_performance=False)
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
