"""Frozen serial profile controller. Failed profiles cannot enter quality/perf."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
import urllib.request
from common import sha, validate_answer, validate_profile, score_letters, product_gate, write
from serial_session import session, memory, info


def request(root, row, rid, budget, *, quality=False):
    parameters = dict(temperature=0, max_new_tokens=budget, ignore_eos=not quality)
    if quality: parameters['stop'] = row['stop']
    payload = dict(rid=rid, input_ids=row['input_ids'], sampling_params=parameters,
                   return_logprob=True, logprob_start_len=-1, top_logprobs_num=0)
    if quality: payload['token_ids_logprob'] = row['candidate_ids']
    value = urllib.request.Request('http://127.0.0.1:31000/generate',
            data=json.dumps(payload).encode(), headers={'Content-Type':'application/json'})
    start = time.monotonic()
    with urllib.request.urlopen(value, timeout=3600) as response: answer = json.load(response)
    seconds = time.monotonic() - start
    ids = validate_answer(answer, rid, row['input_ids'], budget, exact_budget=True)
    with (root / 'requests.jsonl').open('a') as log:
        log.write(json.dumps(dict(payload=payload, response=answer, seconds=seconds)) + '\n')
    return answer, ids, seconds


def check(study, freeze):
    if sha(Path(__file__).with_name('launch_serial.sh')) != freeze['shell_sha256']:
        raise ValueError('LAUNCH_SHELL_PROVENANCE')
    for path, digest in {**freeze['external_tool_hashes'], **freeze['asset_sha256']}.items():
        if sha(path) != digest: raise ValueError('ASSET_OR_TOOL_PROVENANCE:' + path)
    gguf = Path(freeze['gguf'])
    if gguf.stat().st_size != freeze['gguf_stat']['size'] or gguf.stat().st_mtime_ns != freeze['gguf_stat']['mtime_ns']:
        raise ValueError('GGUF_STAT_PROVENANCE')
    for name, digest in freeze['tool_hashes'].items():
        if sha(Path(__file__).parent / name) != digest: raise ValueError('TOOL_PROVENANCE:' + name)
    if sha(study / 'cmmlu.json') != freeze['cmmlu_sha256'] or sha(study / 'compact.json') != freeze['compact_sha256']:
        raise ValueError('CORPUS_PROVENANCE')
    runtime = Path(freeze['runtime_manifest'])
    if sha(runtime) != freeze['runtime_manifest_sha256']: raise ValueError('RUNTIME_MANIFEST_PROVENANCE')
    for row in json.loads(runtime.read_text())['files']:
        if sha(row['path']) != row['sha256']: raise ValueError('RUNTIME_PROVENANCE:' + row['path'])
    for name, row in freeze['profiles'].items():
        profile = study / 'profiles' / (name + '.json')
        if sha(profile) != row['profile_sha256'] or sha(profile.with_suffix('.pt')) != row['pt_sha256']:
            raise ValueError('PROFILE_PROVENANCE')
        validate_profile(json.loads(profile.read_text()))


def route_summary(path, profile):
    data = json.loads(profile.read_text()); ownership = {r['layer']:set(r['cpu_ids']) for r in data['rows']}
    layers, experts, events, mass, tokens, touched, by_layer = set(), set(), 0, 0., 0, 0, {}
    row_keys, touched_keys = {}, set()
    for line in path.read_text().splitlines():
        r = json.loads(line); layer = r['layer']; counts = by_layer.setdefault(layer, dict(rows=0, cpu_events=0, cpu_mass=0., npu_events=0))
        for index, (ids, weights) in enumerate(zip(r['ids'], r['weights'])):
            key = (r['sequence'], index)
            row_keys.setdefault(layer, set()).add(key)
            if len(ids) != 6 or len(weights) != 6: raise ValueError('ROUTE_SHAPE')
            hit = False
            for e, w in zip(ids, weights):
                if e in ownership[layer]:
                    events += 1; mass += w; layers.add(layer); experts.add((layer,e)); hit = True
                    counts['cpu_events'] += 1; counts['cpu_mass'] += w
                else: counts['npu_events'] += 1
            counts['rows'] += 1; tokens += 1; touched += int(hit)
            if hit: touched_keys.add(key)
    if set(by_layer) != set(range(1,27)): raise ValueError('ROUTE_LAYER_COVERAGE')
    if any(row_keys[layer] != row_keys[1] for layer in row_keys): raise ValueError('CROSS_LAYER_ROUTE_CORRELATION')
    if data['resident'] < 64 and len(layers) != 26: raise ValueError('CPU_LAYER_COVERAGE')
    return dict(layers_hit=sorted(layers), experts_hit=sorted(map(list, experts)), cpu_events=events,
        cpu_route_weight_mass=mass, layer_token_rows=tokens, cpu_events_per_layer_token=events/max(tokens,1),
        cpu_touch_fraction_per_layer_token=touched/max(tokens,1), fully_npu_fraction_per_layer_token=1-touched/max(tokens,1),
        scope='FOUR_SHORT_DIAGNOSTIC_REQUESTS_NOT_QUALITY_OR_PERFORMANCE', per_layer=by_layer,
        model_token_rows=len(row_keys[1]), cpu_events_per_model_token=events/max(len(row_keys[1]),1),
        cpu_weight_mass_per_model_token=mass/max(len(row_keys[1]),1),
        token_touches_cpu_fraction=len(touched_keys)/max(len(row_keys[1]),1),
        fully_npu_model_token_fraction=1-len(touched_keys)/max(len(row_keys[1]),1))


def compact(study, profile, target, manifest):
    report = dict(status='RUNNING', references={}, requests=0)
    with session(target / 'compact', profile) as root:
        for row in manifest['prompts']:
            anchor = None
            for i, budget in enumerate(manifest['budgets']):
                _, ids, _ = request(root, row, f'r67-compact-{row["id"]}-{i}', budget)
                if anchor is None: anchor = ids
                if ids != anchor[:budget]: raise ValueError('SAME_PATH_OR_PREFIX_MISMATCH')
                report['requests'] += 1
                write(target / 'compact-progress.json', report)
            report['references'][row['id']] = anchor
    with session(target / 'restart', profile) as root:
        for row in manifest['prompts']:
            _, ids, _ = request(root, row, f'r67-restart-{row["id"]}', 16)
            if ids != report['references'][row['id']]: raise ValueError('CLEAN_RESTART_MISMATCH')
    with session(target / 'observer', profile, observer=True) as root:
        for row in manifest['prompts']:
            _, ids, _ = request(root, row, f'r67-observer-{row["id"]}', 8)
            if ids != report['references'][row['id']][:8]: raise ValueError('OBSERVER_PARITY_MISMATCH')
        report['routing'] = route_summary(root / 'routing.jsonl', profile)
        structure = [json.loads(line) for line in (root / 'physical-structure.jsonl').read_text().splitlines()]
        if {r['layer'] for r in structure} != set(range(1,27)): raise ValueError('PHYSICAL_LAYOUT_INCOMPLETE')
        report['physical_structure'] = structure
        for layer in range(1,27):
            for call in (1,4,7):
                if not (root / 'operands' / f'L{layer:02d}-C{call}.pt').exists(): raise ValueError('REAL_OPERAND_CAPTURE_INCOMPLETE')
    if json.loads(profile.read_text())['resident'] == 64:
        report['local_operator'] = 'N/A_NO_CPU_OWNED_EXPERTS'
    else:
        directory = target / 'operators'; directory.mkdir()
        rows = []
        for layer in range(1,27):
            output = directory / f'L{layer:02d}.json'
            cmd = [sys.executable, str(Path(__file__).with_name('local_operator_screen.py')),
                   '--profile', str(profile), '--operands', str(target / 'observer/operands'),
                   '--layer', str(layer), '--output', str(output)]
            with (directory / f'L{layer:02d}.log').open('x') as log:
                subprocess.run(['taskset', '-c', '0-31', *cmd], stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1800)
            value = json.loads(output.read_text())
            if value['status'] != 'LOCAL_OPERATOR_AND_CONTROL_PASS': raise ValueError('LOCAL_OPERATOR_GATE')
            rows.extend(value['rows'])
            print(json.dumps(dict(profile=profile.stem, phase='LOCAL_OPERATORS', layers=layer, total=26)), flush=True)
        report['local_operator'] = dict(status='PASS', experts=52, operands=len(rows),
            max_rel_l2=max(r['metrics']['relative_l2'] for r in rows), controlled_real_npu_routes=156)
    report['status'] = 'SYSTEM_SCREEN_PASS'
    write(target / 'system.json', report)
    return report


def quality(study, profile, target, inputs):
    report = dict(status='RUNNING', rows=[], cmmlu_sha256=sha(study / 'cmmlu.json'),
                  profile_sha256=sha(profile), count=256, invalid=0)
    with session(target / 'quality', profile) as root:
        for row in inputs['rows']:
            answer, _, seconds = request(root, row, f'r67-quality-{row["id"]}', 1, quality=True)
            values = answer['meta_info']['output_token_ids_logprobs']
            if len(values) != 1: raise ValueError('QUALITY_CANDIDATE_POSITIONS')
            predicted, logits = score_letters(values[0], row['candidate_ids'])
            report['rows'].append(dict(id=row['id'], subject=row['subject'], predicted=predicted,
                correct=predicted==row['answer'], candidate_logprobs=logits, seconds=seconds))
            progress = dict(profile=profile.stem, phase='CMMLU', completed=len(report['rows']), total=256,
                            correct=sum(r['correct'] for r in report['rows']))
            write(target / 'quality-progress.json', progress)
            print(json.dumps(progress), flush=True)
    report.update(status='CMMLU_COLLECTION_COMPLETE', correct=sum(r['correct'] for r in report['rows']))
    write(target / 'quality.json', report)
    return report


def paired(rows, reference):
    ref = {r['id']:r for r in reference}; differences = [int(r['correct'])-int(ref[r['id']]['correct']) for r in rows]
    rng = random.Random(67); simulations = sorted(sum(rng.choice(differences) for _ in differences)/len(differences)*100 for _ in range(20000))
    return dict(regressed=sum(x==-1 for x in differences), improved=sum(x==1 for x in differences),
        delta_pp=statistics.mean(differences)*100, paired_bootstrap_95ci_pp=[simulations[499], simulations[19499]],
        population_noninferiority_claim=False)


def performance(profile, target, manifest):
    sys.path.insert(0, '/workspace/results/r64-migration-tools-v6/tools')
    from stream_timing import request as streaming
    rows = []
    with session(target / 'performance', profile) as root:
        for row in [r for r in manifest['prompts'] if r['id'] in ('s6_en_01','s6_json_01')]:
            reference = None
            for i in range(5):
                result = streaming(f'r67-perf-{row["id"]}-{i}', row['input_ids'], 32)
                ids = validate_answer(result['response'], f'r67-perf-{row["id"]}-{i}', row['input_ids'], 32, True)
                if reference is None: reference = ids
                if ids != reference: raise ValueError('PERFORMANCE_NONDETERMINISM')
                with (root / 'requests.jsonl').open('a') as f: f.write(json.dumps(result)+'\n')
                rows.append(dict(shape=row['id'], warmup=i<2, **{k:v for k,v in result.items() if k!='response'}))
                write(root / f'memory-{row["id"]}-{i}.json', dict(host=memory(), server=info()))
    report = dict(status='ISOLATED_PERFORMANCE_COMPLETE', scope='OBSERVER_OFF_FIXED_NONLOCAL_NUMA_CLIENT_STREAMING', rows=rows,
                  summary={metric:statistics.median(r[metric] for r in rows if not r['warmup']) for metric in
                           ('seconds','ttft_seconds','decode_tokens_per_second')}, cpu_wait=None, host_bandwidth=None,
                  hard_throughput_floor=None)
    write(target / 'performance.json', report)
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--study', type=Path, required=True)
    p.add_argument('--freeze-sha256', required=True)
    a = p.parse_args(); study = a.study
    if not str(study).startswith('/workspace/results/r65-r67-') or sha(study / 'STUDY_FREEZE.json') != a.freeze_sha256:
        raise ValueError('STUDY_FREEZE_PROVENANCE')
    freeze = json.loads((study / 'STUDY_FREEZE.json').read_text())
    check(study, freeze)
    compact_data = json.loads((study / 'compact.json').read_text())
    inputs = json.loads((study / 'cmmlu.json').read_text())
    summary = dict(status='RUNNING', profiles={}, freeze_sha256=a.freeze_sha256, started_unix=time.time(),
                   execution='ONE_EXISTING_CONTAINER_SERIAL', selected_static_baseline=None, round7='NOT_STARTED')
    with (study / 'controller.lock').open('x') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            baseline = None
            for name in freeze['profile_order']:
                check(study, freeze)
                target = study / name; target.mkdir()
                profile = study / 'profiles' / (name + '.json')
                state = summary['profiles'][name] = dict(system='RUNNING', quality='NOT_RUN', performance='NOT_RUN')
                write(study / 'progress.json', summary)
                try:
                    compact(study, profile, target, compact_data)
                    state['system'] = 'PASS'; check(study, freeze)
                    q = quality(study, profile, target, inputs)
                    check(study, freeze)
                    state.update(correct=q['correct'], invalid=q['invalid'])
                    if name == 'R64': baseline = q; state.update(quality='REFERENCE', delta_pp=0)
                    else:
                        if baseline is None: raise ValueError('MISSING_FRESH_REFERENCE')
                        verdict = product_gate(q['correct'], baseline['correct'])
                        state.update(quality='PASS' if verdict['pass_5pp'] else 'FAIL_5PP', delta_pp=verdict['delta_pp'],
                                     paired=paired(q['rows'], baseline['rows']))
                except Exception as exc:
                    check(study, freeze)
                    state.update(system='FAIL', error=repr(exc), quality='NOT_QUALIFIED_SYSTEM_FAILURE')
                    write(target / 'failure.json', state)
                    if name == 'R64': raise RuntimeError('FRESH_BASELINE_SYSTEM_FAILURE') from exc
                write(study / 'progress.json', summary)
            for name, state in summary['profiles'].items():
                if state['system'] == 'PASS' and state['quality'] in ('PASS','REFERENCE'):
                    check(study, freeze)
                    try:
                        perf = performance(study / 'profiles' / (name + '.json'), study / name, compact_data)
                        state.update(performance='COMPLETE', performance_summary=perf['summary'])
                    except Exception as exc:
                        state.update(performance='FAILED', performance_error=repr(exc))
                        state.update(system='FAIL_AFTER_PERFORMANCE_EXCEPTION', quality='NOT_QUALIFIED_LATE_SYSTEM_FAILURE')
                    check(study, freeze); write(study / 'progress.json', summary)
            candidates = [name for name,state in summary['profiles'].items() if name != 'R64' and
                          state['system'] == 'PASS' and state['quality'] == 'PASS' and state['performance'] == 'COMPLETE']
            summary.update(status='SERIAL_SWEEP_COMPLETE', accepted_candidates=candidates,
                           selection_status='PARETO_REVIEW_REQUIRED' if candidates else 'NO_QUALIFIED_HEAVY_BASELINE')
        except BaseException as exc:
            summary.update(status='FAILED', error=repr(exc)); raise
        finally:
            summary['ended_unix'] = time.time(); write(study / 'summary.json', summary)
            write(study / 'progress.json', summary)
            artifacts = {str(p.relative_to(study)):sha(p) for p in sorted(study.rglob('*')) if p.is_file() and
                         p.suffix in ('.json','.jsonl','.pt') and p.name not in ('artifact_hashes.json','progress.json')}
            write(study / 'artifact_hashes.json', artifacts)


if __name__ == '__main__': main()
