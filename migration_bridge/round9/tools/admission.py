"""Fail-closed physical15 admission evaluator. Never launches or reconfigures devices.

Inputs are measured evidence, not resource reservations. PASS applies only to
the recorded budget and timestamp; it cannot predict another job's growth.
"""
import argparse
import json
import math
import pathlib
import time


def evaluate(snapshot, now):
    errors = []
    required = {'physical_id', 'device_nodes', 'mapping_verified',
                'access_verified', 'identity_status', 'prerequisite_statuses',
                'sampled_at', 'hbm_total_mib', 'training_peak_mib',
                'owned_peak_budget_mib', 'reserve_mib', 'cpu_physical_keys',
                'cpu_set', 'observed_cpu_keys', 'telemetry_ok'}
    if not isinstance(snapshot, dict) or not required <= snapshot.keys():
        return {'status': 'BLOCKED', 'reasons': ['missing admission evidence']}
    if snapshot['physical_id'] != 15 or snapshot['device_nodes'] != [15]:
        errors.append('only physical15 may be exposed')
    for name in ('mapping_verified', 'access_verified', 'telemetry_ok'):
        if snapshot[name] is not True:
            errors.append(name + ' is not verified')
    if snapshot['identity_status'] != 'PASS':
        errors.append('source identity not PASS')
    deps = snapshot['prerequisite_statuses']
    if (not isinstance(deps, dict) or deps.get('runtime') != 'PASS'
            or any(v != 'PASS' for v in deps.values())):
        errors.append('prerequisites not PASS')
    fields = ('sampled_at', 'hbm_total_mib', 'training_peak_mib',
              'owned_peak_budget_mib', 'reserve_mib')
    numeric = all(type(snapshot[k]) in (int, float) and
                  math.isfinite(snapshot[k]) and snapshot[k] >= 0 for k in fields)
    if not numeric:
        errors.append('invalid telemetry or budget numbers')
    else:
        if not 0 <= now - snapshot['sampled_at'] <= 30:
            errors.append('telemetry must be no more than30s old and not in future')
        if snapshot['hbm_total_mib'] != 65536:
            errors.append('unexpected device capacity')
        if snapshot['owned_peak_budget_mib'] <= 0 or snapshot['reserve_mib'] < 8192:
            errors.append('invalid owned budget or reserve below8GiB')
        if sum(snapshot[k] for k in ('training_peak_mib', 'owned_peak_budget_mib',
                                     'reserve_mib')) > snapshot['hbm_total_mib']:
            errors.append('HBM budget exceeds physical capacity')
    cpus, keys, topology = (snapshot['cpu_set'], snapshot['cpu_physical_keys'],
                            snapshot['observed_cpu_keys'])
    if (not isinstance(cpus, list) or not isinstance(keys, list)
            or not isinstance(topology, dict) or not cpus
            or any(type(c) is not int or c < 0 for c in cpus)
            or any(not isinstance(k, str) for k in keys)):
        errors.append('invalid CPU topology evidence')
    elif (len(set(cpus)) != len(cpus) or len(set(keys)) != len(keys)
          or len(keys) != len(cpus)
          or [topology.get(str(c)) for c in cpus] != keys):
        errors.append('duplicate SMT core or CPU mapping mismatch')
    return {'status': 'BLOCKED' if errors else 'PASS', 'reasons': errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot', type=pathlib.Path)
    args = parser.parse_args()
    result = evaluate(json.loads(args.snapshot.read_text()), time.time())
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['status'] == 'PASS' else 2)


if __name__ == '__main__':
    main()
