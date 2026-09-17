"""Decode-only exposure, rejecting unassociated/incomplete model forwards.

Prefill routed ownership is not actual CPU execution when Dynamic enters the
full-device path. Keep it out of the post-update Hybrid decode comparison.
The initial-mask lane is a counterfactual on the SAME Dynamic routes; it must
not replace measured Static-vs-Dynamic results, whose routes can differ.
"""


def correlate_updates(updates,requests):
    groups={r['id']:[] for r in requests}
    for event in updates:
        matched=[r['id'] for r in requests
                 if r['started_unix']<=event['time_unix']<=r['ended_unix']]
        if len(matched)!=1:raise ValueError('UNASSOCIATED_OR_AMBIGUOUS_MIGRATION_EVENT')
        groups[matched[0]].append(event)
    triggered=sum(bool(events) for events in groups.values())
    return dict(requests=len(requests),triggered_requests=triggered,
        actual_trigger_rate=triggered/len(requests),
        update_calls=len(updates),copied_expert_slots=sum(len(e['selected']) for e in updates),
        changed_resident_experts=sum(len(set(e['selected'])-set(e['previous_resident'])) for e in updates),
        migration_bytes=sum(e['migration_bytes'] for e in updates),
        per_request=[dict(id=rid,updated_layers=[e['layer'] for e in events]) for rid,events in groups.items()],
        scope='ACTUAL_UPDATE_EVENTS_ASSOCIATED_TO_NONOVERLAPPING_SERIAL_REQUEST_WINDOWS')


def summarize_decode(rows,kept):
    result=dict(assignments=0,npu_hits=0,cpu_hits=0,total_weight=0.0,cpu_weight=0.0,
        initial_static_npu_hits=0,initial_static_cpu_weight=0.0)
    groups={}
    for r in rows:
        context=r['forward_context']
        key=(context['epoch'],context['forward_id'])
        if key not in kept:continue
        if 'DECODE' not in context['mode'].upper():continue
        layers=groups.setdefault(key,{})
        if r['layer'] in layers:raise ValueError('DUPLICATE_DECODE_LAYER')
        layers[r['layer']]=r
        for field in result:result[field]+=r[field]
    if not groups:raise ValueError('NO_ATTRIBUTED_DECODE_FORWARD')
    token_rows=0;fully_npu_rows=0
    for layers in groups.values():
        if set(layers)!=set(range(1,27)):raise ValueError('DECODE_LAYER_INCOMPLETENESS')
        sizes={r['tokens'] for r in layers.values()}
        if len(sizes)!=1:raise ValueError('DECODE_ROW_SHAPE_DRIFT')
        count=next(iter(sizes));token_rows+=count
        for r in layers.values():
            if len(r['fully_npu_row_flags'])!=count or r['assignments']!=6*count:
                raise ValueError('DECODE_ROUTE_SHAPE_DRIFT')
        fully_npu_rows+=sum(all(r['fully_npu_row_flags'][i] for r in layers.values())
                           for i in range(count))
    if result['assignments']!=token_rows*26*6:raise ValueError('DECODE_ROUTE_COMPLETENESS')
    result.update(npu_hit_rate=result['npu_hits']/result['assignments'],
        cpu_route_rate=result['cpu_hits']/result['assignments'],
        cpu_weight_mass=result['cpu_weight']/result['total_weight'],
        initial_static_npu_hit_rate=result['initial_static_npu_hits']/result['assignments'],
        initial_static_cpu_weight_mass=result['initial_static_cpu_weight']/result['total_weight'],
        model_token_rows=token_rows,fully_npu_model_token_rows=fully_npu_rows,
        model_token_touches_cpu_fraction=1-fully_npu_rows/token_rows,
        counterfactual_scope='INITIAL_STATIC_MASK_ON_SAME_DYNAMIC_ROUTES_NOT_SEPARATE_STATIC_RUN',
        scope='ATTRIBUTED_HYBRID_DECODE_ONLY_PREFILL_AND_UNUSED_TAIL_EXCLUDED')
    return result
