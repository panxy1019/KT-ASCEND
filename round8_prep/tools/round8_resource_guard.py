"""Fail-closed offline admission check; input must be fresh measured inventory.
No process is started, killed, or reconfigured by this tool.
"""
import argparse
import json

def reject_reasons(x):
    reasons = []
    required = ['inventory_age_seconds','round7_perf_active','round7_npus','round7_cpus',
                'requested_npu','requested_cpus','host_available_bytes','host_total_bytes',
                'predicted_private_bytes','swap_used_bytes','npu_free_bytes',
                'predicted_npu_peak_bytes','disk_free_bytes','download_conversion_budget_bytes',
                'other_npu_processes','same_host_inventory','physical_core_count']
    missing = [k for k in required if k not in x]
    if missing: return ['MISSING:'+','.join(missing)]
    if not x['same_host_inventory']: reasons.append('WRONG_HOST_INVENTORY')
    if not 0 <= x['inventory_age_seconds'] <= 60: reasons.append('STALE_INVENTORY')
    if x['round7_perf_active']: reasons.append('ROUND7_PERF_ACTIVE')
    if x['requested_npu'] in x['round7_npus'] or x['other_npu_processes']: reasons.append('NPU_BUSY')
    cpus = x['requested_cpus']
    if len(set(cpus)) != 64 or x['physical_core_count'] != 64: reasons.append('C64_NOT_PROVEN')
    if set(cpus) & set(x['round7_cpus']): reasons.append('CPU_COLLISION')
    reserve = max(64*2**30, x['host_total_bytes']*.2)
    if x['host_available_bytes']-x['predicted_private_bytes'] < reserve: reasons.append('RAM_RESERVE')
    if x['swap_used_bytes'] != 0:
        # Residual unrelated host swap is not evidence of current paging.
        # Allow bounded readback of old unrelated pages, never new swap-out.
        # Byte-based (not page-based) so 4KiB/64KiB hosts have the same cap.
        readback=x.get('host_swapin_delta',-1)
        page_size=x.get('host_page_size_bytes',0)
        idle = (x.get('swap_observation_seconds', 0) >= 30
                and isinstance(readback,int) and readback >= 0
                and (readback == 0 or (page_size > 0 and readback*page_size <= 2**20))
                and x.get('host_swapout_delta') == 0
                and x.get('cgroup_swap_current') == 0)
        if not idle: reasons.append('SWAP_ACTIVE_OR_UNVERIFIED')
    if x['npu_free_bytes'] < x['predicted_npu_peak_bytes'] + 4*2**30: reasons.append('HBM_RESERVE')
    if x['disk_free_bytes'] < x['download_conversion_budget_bytes']: reasons.append('DISK_BUDGET')
    return reasons

if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('inventory'); a=p.parse_args()
    reasons=reject_reasons(json.load(open(a.inventory)))
    print(json.dumps(dict(admitted=not reasons,reasons=reasons)))
    raise SystemExit(bool(reasons))
