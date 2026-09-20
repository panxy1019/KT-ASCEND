"""Audit contiguous frozen reference slices without inheriting another score."""
import argparse,hashlib,json,math,pathlib,re
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=pathlib.Path,required=True);p.add_argument('--slices',type=pathlib.Path,nargs='+',required=True);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
manifest=json.loads(a.manifest.read_text());digest=hashlib.sha256(a.manifest.read_bytes()).hexdigest()
assert digest=='88efae7d68e7edf644add12909db89712dc001046b8624801dbb8ce7a8804858'
assert manifest['revision']=='de4b8e4d43b917e7706784d8bb445c9af86a3540' and len(manifest['rows'])==256
round9=pathlib.Path(__file__).parents[1]
expected_driver=hashlib.sha256((round9/'tests/full_model_cmmlu.py').read_bytes()).hexdigest()
expected_profile='2a88116fb6a67f31149494104106857a88ba4fcf4078a6b2606b3d1fe64e69c4'
assert hashlib.sha256((round9/'runtime/profiles/q64_bf16_c64_sync_physical15.json').read_bytes()).hexdigest()==expected_profile
assert len(a.slices)==2
records=[];ranges=[];next_start=0;driver=None;runtime=None;container_ids=[]
for slice_index,root in enumerate(a.slices):
 summary=json.loads((root/'summary.json').read_text());assert summary['status']=='REFERENCE_SLICE_SCORED_PENDING_AUDIT'
 assert summary['manifest_sha256']==digest and summary['invalid']==0
 assert summary['start']==next_start==slice_index*128 and summary['count']==128
 assert hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()==digest
 assert summary['freeze_ack']['manifest_sha256']==digest and summary['freeze_ack']['local_commit']=='e66726f2f6a07aad85a974a15e0aba7604a20852'
 assert re.fullmatch('[0-9a-f]{64}',summary['driver_sha256']) and re.fullmatch('[0-9a-f]{64}',summary['restart_artifact_sha256'])
 assert summary['driver_sha256']==expected_driver
 run=root.parent
 admission=json.loads((run/'preflight/admission.json').read_text())
 container=json.loads((run/'container-created.json').read_text())
 restart_path=run/'observer-off-v1/summary.json'
 restart=json.loads(restart_path.read_text())
 assert hashlib.sha256(restart_path.read_bytes()).hexdigest()==summary['restart_artifact_sha256']
 assert admission['status'] in ['PASS','STATIC_BOUND_READY_FOR_MONITORED_FULL_MODEL_RUN']
 assert re.fullmatch('[0-9a-f]{64}',admission['implementation_sha256'])
 assert admission['profile_sha256']==expected_profile
 assert [v for v in container['Config']['Env'] if v.startswith('ROUND9_MEMORY_OBSERVER=')]==['ROUND9_MEMORY_OBSERVER=0']
 assert re.fullmatch('sha256:[0-9a-f]{64}',container['Image'])
 assert restart['status']=='PASS_RESTART4' and restart['container_id']==container['Id']==summary['container_id']
 assert restart['implementation_sha256']==admission['implementation_sha256']
 assert restart['profile_sha256']==admission['profile_sha256']
 expected_runtime=dict(implementation_sha256=admission['implementation_sha256'],profile_sha256=admission['profile_sha256'],image=container['Image'],observer=False,admission_status=admission['status'])
 assert summary['runtime_identity']==expected_runtime
 container_ids.append(container['Id'])
 driver=driver or summary['driver_sha256'];assert summary['driver_sha256']==driver
 runtime=runtime or summary['runtime_identity'];assert summary['runtime_identity']==runtime and runtime['observer'] is False
 assert runtime['admission_status'] in ['PASS','STATIC_BOUND_READY_FOR_MONITORED_FULL_MODEL_RUN']
 assert re.fullmatch('[0-9a-f]{64}',runtime['implementation_sha256'])
 assert runtime['profile_sha256']=='2a88116fb6a67f31149494104106857a88ba4fcf4078a6b2606b3d1fe64e69c4'
 assert re.fullmatch('sha256:[0-9a-f]{64}',runtime['image'])
 rows=[json.loads(l) for l in (root/'raw.jsonl').read_text().splitlines()]
 assert len(rows)==summary['count'];records.extend(rows);ranges.append([next_start,next_start+len(rows)]);next_start+=len(rows)
assert len(set(container_ids))==2
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
