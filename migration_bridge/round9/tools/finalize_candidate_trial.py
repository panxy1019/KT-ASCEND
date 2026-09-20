"""Bind completed functional/resource evidence; never grants accuracy qualification."""
import argparse,hashlib,json,pathlib,subprocess
def classify_failures(failed):
 resource_failed=[name for name in failed if name!='canaries.json']
 if resource_failed:return 'FAIL_CANDIDATE_RESOURCE_TRIAL','FAIL_CANDIDATE_RESOURCE_SYSTEM'
 if failed:return 'FAIL_CANDIDATE_CONTENT_CANARY','PASS_CANDIDATE_RESOURCE_SYSTEM'
 return 'PASS_CANDIDATE_SYSTEM_RESOURCE_ONLY','PASS_CANDIDATE_RESOURCE_SYSTEM'
def content_only_failure(report):
 rows=report.get('rows',[])
 if not rows or not all(r.get('numeric_finite') is True and r.get('response',{}).get('text','').strip() for r in rows):return False
 last=rows[-1];error=report.get('error','')
 return report.get('status')=='FAIL' and last.get('passed') is False and (last.get('parse_error')==error or error.startswith('AssertionError('))
p=argparse.ArgumentParser();p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--container',required=True);a=p.parse_args()
root=a.output
if (root/'trial-summary.json').exists():raise FileExistsError(root/'trial-summary.json')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
required={'first-http.json':'PASS_FIRST_REAL_HTTP_PREFILL_DECODE','system8.json':'PASS_FULL_MODEL_SYSTEM8',
 'canaries.json':'PASS_FOUR_KNOWN_ANSWER_CANARIES','context4096.json':'PASS_4096_TOKEN_RESOURCE_REQUEST',
 'cancel-reuse.json':'PASS_ACTIVE_ABORT_AND_STATE_REUSE'}
results={name:json.loads((root/name).read_text()) for name in required}
failed=[name for name,status in required.items() if results[name].get('status')!=status]
if 'canaries.json' in failed and not content_only_failure(results['canaries.json']):failed.append('canary_technical_or_numeric_failure')
state=json.loads(subprocess.check_output(['docker','inspect',a.container],text=True))[0]
created=json.loads((root/'container-created.json').read_text())
if state['Id']!=created['Id']:raise RuntimeError('container identity changed')
if state['State']['Running']:raise RuntimeError('stop the completed owned trial before finalizing immutable log hashes')
rows=[json.loads(line) for line in (root/'monitor.jsonl').read_text().splitlines()]
if not rows or rows[-1].get('action') not in ('OWNED_EXITED','OWNED_EXITED_DURING_SAMPLE'):raise RuntimeError('wait for monitor final exit record')
if any(row.get('action')=='STOP_OWNED' for row in rows):failed.append('guard_stopped_run')
if state['State']['OOMKilled']:failed.append('container_OOMKilled')
for row in rows:
 cg=row.get('owned_cgroup',{})
 if int(cg.get('memory.failcnt','0')):failed.append('cgroup_memory_failure');break
stop=json.loads((root/'controlled-stop.json').read_text())
inference_rows=[r for r in rows if r['time']<stop['time']]
overall,resource_status=classify_failures(failed)
summary=dict(telemetry_scope='before explicitly recorded controlled shutdown; old guard could read root cgroup during exit',status=overall,
 resource_system_status=resource_status,content_canary_status=results['canaries.json']['status'],
 scope='FUNCTIONAL_RESOURCE_NOT_ACCURACY_PERFORMANCE_OR_DEPLOYMENT_QUALIFICATION',failed=failed,
 container_id=state['Id'],resource_mode=json.loads((root/'admission.json').read_text())['resource_mode'],
 evidence_sha256={name:sha(root/name) for name in list(required)+['admission.json','payload-sha256.json','server-profile.json','resolved-profile.json','monitor.jsonl','server.log','controlled-stop.json']},
 peak_observed_board_mib=max(r.get('hbm_used_mib',0) for r in inference_rows),peak_observed_owned_mib=max(r.get('owned_process_mib',0) for r in inference_rows),
 peak_observed_host_cgroup_bytes=max(int(r.get('owned_cgroup',{}).get('memory.max_usage_in_bytes','0')) for r in inference_rows),
 same_placement_static_oracle='PENDING_FOR_DYNAMIC',quality_reference256='INCOMPLETE_NO_INHERITANCE',
 restart_qualification='PENDING')
# Bind the final request-boundary log, including context and abort transactions
# that occur after the system8 report's embedded event snapshot.
for name in ('dynamic-events.jsonl','scheduler-affinity.json'):
 if (root/name).exists():summary['evidence_sha256'][name]=sha(root/name)
summary['finalizer_sha256']=sha(pathlib.Path(__file__))
(root/'trial-summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if failed:raise SystemExit(1)
