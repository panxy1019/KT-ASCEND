"""Post-collection capacity/quality/Pareto report. No inference or threshold fit."""
import argparse
import json
from pathlib import Path
import statistics
from common import sha, write


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--study',type=Path,required=True)
    a=p.parse_args(); root=a.study
    summary=json.loads((root/'summary.json').read_text())
    if summary['status']!='SERIAL_SWEEP_COMPLETE':
        raise ValueError('COLLECTION_NOT_COMPLETE')
    freeze=json.loads((root/'STUDY_FREEZE.json').read_text())
    if sha(root/'STUDY_FREEZE.json')!=summary['freeze_sha256']:
        raise ValueError('FREEZE_IDENTITY')
    telemetry=[]
    for path in (root/'host-telemetry.jsonl',root/'host-telemetry-v2.jsonl'):
        if path.exists():
            for line in path.read_text().splitlines():
                try:telemetry.append(json.loads(line))
                except json.JSONDecodeError:pass  # in-progress final sampler line only
    telemetry.sort(key=lambda x:x['time_unix'])
    results={};baseline=summary['profiles']['R64']['correct']
    for name,state in summary['profiles'].items():
        result=dict(state, resident=int(name[1:]), theoretical_routed_bf16_gib=26.8125*int(name[1:])/64,
                    hbm_sessions={}, shared_cache_not_private_ram=True)
        for phase in ('compact','restart','observer','quality','performance'):
            lifecycle=root/name/phase/'lifecycle.json'
            if not lifecycle.exists():continue
            life=json.loads(lifecycle.read_text());start=life['started_unix'];end=life['ended_unix']
            sampled=[r for r in telemetry if start<=r['time_unix']<=end and r.get('npu0_hbm') is not None]
            before=[r for r in telemetry if r['time_unix']<=start and r.get('npu0_hbm') is not None]
            ready=root/name/phase/'ready.json';ready_time=ready.stat().st_mtime if ready.exists() else None
            after=[r for r in sampled if ready_time is not None and r['time_unix']>=ready_time]
            hbm=dict(before_load_mib=before[-1]['npu0_hbm']['used_mib'] if before else None,
                after_load_mib=after[0]['npu0_hbm']['used_mib'] if after else None,
                sampled_peak_mib=max((r['npu0_hbm']['used_mib'] for r in sampled),default=None),
                steady_median_mib=statistics.median(r['npu0_hbm']['used_mib'] for r in after) if after else None,
                sample_interval_seconds=10, exact_peak_claim=False,
                peak_container_process_rss_sum_bytes=max((sum(x['memory'].get('VmRSS') or 0 for x in r['processes']) for r in sampled),default=None),
                peak_container_process_anon_sum_bytes=max((sum(x['memory'].get('RssAnon') or 0 for x in r['processes']) for r in sampled),default=None),
                peak_container_process_file_sum_bytes=max((sum(x['memory'].get('RssFile') or 0 for x in r['processes']) for r in sampled),default=None),
                rss_file_sum_can_double_count=True,
                host_mem_available_min_bytes=min((r['host_memory']['MemAvailable'] for r in sampled),default=None))
            result['hbm_sessions'][phase]=hbm
        system=root/name/'system.json'
        if system.exists():
            data=json.loads(system.read_text());result['routing']=data['routing'];result['local_operator']=data['local_operator']
        footprint=result['hbm_sessions'].get('performance',{}).get('steady_median_mib')
        if footprint is None:footprint=result['hbm_sessions'].get('quality',{}).get('steady_median_mib')
        result['observed_device_steady_mib']=footprint
        results[name]=result
    candidates=[n for n in summary['accepted_candidates'] if results[n]['observed_device_steady_mib'] is not None]
    pareto=[]
    for name in candidates:
        x=results[name];bw=x['performance_summary']['decode_tokens_per_second'];hbm=x['observed_device_steady_mib']
        dominated=any(results[o]['observed_device_steady_mib']<=hbm and
            results[o]['performance_summary']['decode_tokens_per_second']>=bw and
            (results[o]['observed_device_steady_mib']<hbm or results[o]['performance_summary']['decode_tokens_per_second']>bw)
            for o in candidates if o!=name)
        if not dominated:pareto.append(name)
    selected=min(pareto,key=lambda n:(results[n]['observed_device_steady_mib'],-results[n]['performance_summary']['decode_tokens_per_second'])) if pareto else None
    report=dict(version='r67-serial-product-results-v1',freeze_sha256=summary['freeze_sha256'],
        baseline_correct=baseline, profiles=results, pareto_profiles=pareto,
        selected_capacity_first_entry_candidate=selected, selection_rule='LOWEST_MEASURED_HBM_ON_ACCEPTED_PARETO_NO_SPEED_FLOOR',
        interactive_throughput_requirement='NOT_SPECIFIED_REVIEW_TELEMETRY',
        static_frequency='NOT_STARTED',round7='NOT_STARTED',strict_numerical_equivalence='NOT_QUALIFIED',
        summarizer_sha256=sha(__file__))
    write(root/'pareto_summary.json',report)
    lines=['# Round6.7 serial all-MoE product screen','','| Profile | Correct /256 | Delta pp | System | CMMLU | Steady HBM MiB | Decode tok/s |',
           '|---|---:|---:|---|---|---:|---:|']
    for n,r in results.items():
        speed=r.get('performance_summary',{}).get('decode_tokens_per_second')
        lines.append(f'|{n}|{r.get("correct","N/A")}|{r.get("delta_pp","N/A")}|{r["system"]}|{r["quality"]}|{r["observed_device_steady_mib"]}|{speed}|')
    lines+=['',f'Capacity-first Static Frequency entry candidate: {selected or "NONE"}.',
        f'Accepted Pareto profiles: {pareto}.',
        'No throughput floor was pre-registered; this selection is not a claim of interactive usability.',
        'The 5pp observed CMMLU screen is not population non-inferiority or strict numerical equivalence.',
        'Fixed CPU NUMA0 / NPU NUMA6 is nonlocal; host bandwidth and CPU-wait counters were not measured.',
        'Static Frequency and Round7 Dynamic remain NOT_STARTED. Deferred remains OFF.','']
    (root/'ROUND6_7_SERIAL_RESULTS.md').write_text('\n'.join(lines))
    print(json.dumps(dict(status='POST_COLLECTION_REVIEW_COMPLETE',selected=selected,pareto=pareto)))


if __name__=='__main__':main()
